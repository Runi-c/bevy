//! A fine-grained reactivity system built on entities and change detection.
//!
//! This module provides effects (reactive computations that re-run when the data they read
//! changes) and [`Signal`]s (reactive values for state that isn't already a component). Both are
//! backed by entities, so they participate in normal [`World`] storage, despawning, and
//! relationships.
//!
//! ## Sources
//!
//! A reactive *source* is **any component on any entity**, or **any resource**. Effects track them
//! with [`EffectContext::track_component`] / [`EffectContext::track_resource`], and ordinary
//! systems update them by ordinary mutation. No special setter API is required:
//!
//! ```
//! # use bevy_ecs::prelude::*;
//! # #[derive(Component)] struct Health { current: u32 }
//! fn damage(mut query: Query<&mut Health>) {
//!     for mut health in &mut query {
//!         health.current -= 1; // an effect tracking this entity's `Health` will re-run
//!     }
//! }
//! ```
//!
//! [`Signal<T>`] is the special case where the source is a dedicated entity holding a
//! [`SignalValue<T>`]. Use it for reactive state that isn't naturally a component, or where you
//! want a source finer than one component (change detection is per-component, so tracking a
//! component means any write to *any* of its fields re-runs the effect).
//!
//! ## Dependency tracking
//!
//! Dependencies are tracked _dynamically_: while an effect body runs, each `track` call records an
//! edge. An effect therefore subscribes only to the sources it actually read on its most recent
//! run, and conditional reads subscribe/unsubscribe as conditions change. Edges are stored on both
//! ends — [`Subscribers`] on the source entity, [`Dependencies`] on the effect entity — and are
//! reconciled after every run.
//!
//! ## Push/pull and the tick invariant
//!
//! Updates are push/pull, using change ticks as the version counter:
//!
//! - **Push**: [`poll_sources`] scans tracked sources and queues the subscribers of any whose
//!   `changed` tick advanced. This only marks effects as *maybe*-dirty; nothing is recomputed.
//! - **Pull**: before running a queued effect, each of its dependencies' current `changed` tick is
//!   compared against the tick that effect *observed* on its previous run. If nothing advanced,
//!   the run is skipped.
//!
//! Within a flush, effects run in [`EffectIndex`] order — registration order — rather than in
//! whatever order they happened to be marked dirty, so a flush is deterministic.
//!
//! The observed tick is captured **after** the effect body completes, not at read time. That is
//! what makes self-cycles structurally impossible: an effect that writes a component it also reads
//! has, by construction, already observed the consequences of its own write, so its own write can
//! never re-trigger it. Unlike suppressing change detection on effect writes, this keeps chained
//! derivations working — another effect tracking that component still sees the change.
//!
//! Note that [`World::get_mut`] does *not* advance the world's change tick, so the tick is
//! incremented explicitly around each effect run and each poll. This gives every effect run its
//! own tick epoch, exactly as the schedule executor does for each system. Without it, every write
//! inside a single flush would be stamped identically and the comparisons above would be
//! meaningless.
//!
//! ```
//! # use bevy_ecs::prelude::*;
//! # use bevy_ecs::signal::{Signal, spawn_effect};
//! #[derive(Component, Default, Clone)]
//! struct Doubled(u32);
//!
//! let mut world = World::new();
//! let count = world.spawn_signal(1u32);
//! let target = world.spawn(Doubled::default()).id();
//!
//! spawn_effect(&mut world, move |ctx| {
//!     let value = ctx.track(count).unwrap_or_default();
//!     if let Some(mut doubled) = ctx.world().get_mut::<Doubled>(target) {
//!         doubled.0 = value * 2;
//!     }
//! });
//! assert_eq!(world.get::<Doubled>(target).unwrap().0, 2);
//!
//! *world.signal_mut(count).unwrap() = 5;
//! world.run_effects();
//! assert_eq!(world.get::<Doubled>(target).unwrap().0, 10);
//! ```

use crate::{
    change_detection::{Mut, Tick},
    component::{Component, ComponentId},
    entity::Entity,
    lifecycle::HookContext,
    resource::Resource,
    world::{DeferredWorld, World},
};
use alloc::{boxed::Box, sync::Arc, vec::Vec};
use bevy_platform::collections::{HashMap, HashSet};
use core::marker::PhantomData;

mod field_effect;
pub mod source;

pub use field_effect::{DedupedFieldEffect, FieldEffect, LastValue, QueuedEffects};

/// The maximum number of times [`World::run_effects`] will poll-and-drain in a single flush before
/// giving up. This is a backstop against genuine oscillation (effects whose values keep actually
/// changing); ordinary self-writes are handled by the tick invariant, not by this limit.
pub const MAX_EFFECT_ITERATIONS: usize = 128;

/// A reactive source: something whose change can wake an effect.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum Source {
    /// A specific component on a specific entity.
    Component {
        /// The entity holding the component.
        entity: Entity,
        /// The component being tracked.
        component: ComponentId,
    },
    /// A resource. Resources have change ticks too, so they slot into the same machinery — the
    /// only difference is where the subscriber list lives, since there is no entity to hang it on.
    Resource {
        /// The resource being tracked, by its [`ComponentId`].
        resource: ComponentId,
    },
}

/// A dependency recorded by an effect: a [`Source`], plus the `changed` tick that source had when
/// the effect *finished* its last run.
///
/// See the module docs for why the tick is captured after the run rather than at read time.
#[derive(Copy, Clone, Debug)]
pub struct Dependency {
    /// The tracked source.
    pub source: Source,
    /// The source's `changed` tick as of the end of the effect's last run.
    pub observed: Tick,
}

/// A handle to a reactive value stored on a dedicated entity.
///
/// This is a convenience over the general `(Entity, ComponentId)` source model, for reactive state
/// that isn't naturally a component. The value lives in a [`SignalValue<T>`] component on
/// [`Signal::entity`]. This handle is [`Copy`] regardless of whether `T` is, so it can be freely
/// captured by effect closures.
pub struct Signal<T> {
    entity: Entity,
    marker: PhantomData<fn() -> T>,
}

impl<T> Clone for Signal<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for Signal<T> {}

impl<T> core::fmt::Debug for Signal<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_tuple("Signal").field(&self.entity).finish()
    }
}

impl<T> PartialEq for Signal<T> {
    fn eq(&self, other: &Self) -> bool {
        self.entity == other.entity
    }
}

impl<T> Eq for Signal<T> {}

impl<T: Send + Sync + 'static> Signal<T> {
    /// Creates a handle for an existing signal entity.
    ///
    /// The caller is responsible for ensuring `entity` actually holds a [`SignalValue<T>`].
    pub fn from_entity(entity: Entity) -> Self {
        Self {
            entity,
            marker: PhantomData,
        }
    }

    /// The entity backing this signal.
    pub fn entity(self) -> Entity {
        self.entity
    }
}

/// The component holding a [`Signal`]'s current value.
#[derive(Component)]
pub struct SignalValue<T: Send + Sync + 'static>(pub T);

/// The effects subscribed to each tracked component of this entity.
///
/// This is the inverse of [`Dependencies`]; both sides are maintained together. It is inserted
/// automatically on a source entity the first time one of its components is tracked.
#[derive(Component, Default, Debug)]
pub struct Subscribers(HashMap<ComponentId, Vec<Entity>>);

impl Subscribers {
    /// The effects subscribed to the given component of this entity.
    pub fn get(&self, component: ComponentId) -> &[Entity] {
        self.0.get(&component).map_or(&[], Vec::as_slice)
    }

    /// Returns true if no effects are subscribed to any component of this entity.
    pub fn is_empty(&self) -> bool {
        self.0.values().all(Vec::is_empty)
    }
}

/// The sources an effect read on its most recent run, and the ticks it observed for them.
///
/// This is the inverse of [`Subscribers`]. When this component is removed (including when the
/// effect entity is despawned), the effect is unsubscribed from each of its sources.
#[derive(Component, Default, Debug)]
#[component(on_remove = Dependencies::on_remove)]
pub struct Dependencies(Vec<Dependency>);

impl Dependencies {
    /// The dependencies recorded on the effect's last run.
    pub fn iter(&self) -> impl Iterator<Item = Dependency> + '_ {
        self.0.iter().copied()
    }

    fn on_remove(mut world: DeferredWorld, context: HookContext) {
        let Some(dependencies) = world
            .get::<Dependencies>(context.entity)
            .map(|deps| deps.0.clone())
        else {
            return;
        };
        for dependency in dependencies {
            match dependency.source {
                Source::Component { entity, component } => {
                    if let Some(mut subscribers) = world.get_mut::<Subscribers>(entity)
                        && let Some(subs) = subscribers.0.get_mut(&component)
                    {
                        subs.retain(|effect| *effect != context.entity);
                    }
                }
                Source::Resource { resource } => {
                    if let Some(mut subscribers) = world.get_resource_mut::<ResourceSubscribers>()
                        && let Some(subs) = subscribers.0.get_mut(&resource)
                    {
                        subs.retain(|effect| *effect != context.entity);
                    }
                }
            }
        }
    }
}

/// A reactive computation. The boxed function re-runs when a source it tracked changes.
#[derive(Component, Clone)]
pub struct Effect(pub Arc<dyn Fn(&mut EffectContext) + Send + Sync>);

/// Relates an effect entity to the entity whose lifetime it is tied to.
///
/// Effects spawned with [`spawn_effect_for`] use this so they are despawned along with their
/// target, rather than leaking once the thing they were updating is gone.
#[derive(Component, Debug)]
#[relationship(relationship_target = Effects)]
pub struct EffectOf(pub Entity);

/// The effects whose lifetime is tied to this entity. See [`EffectOf`].
#[derive(Component, Debug)]
#[relationship_target(relationship = EffectOf, linked_spawn)]
pub struct Effects(Vec<Entity>);

/// The effects subscribed to each tracked resource.
///
/// The resource-side counterpart of [`Subscribers`]. Resources have no entity to store a component
/// on, so their subscriber lists live here.
#[derive(Resource, Default, Debug)]
pub struct ResourceSubscribers(HashMap<ComponentId, Vec<Entity>>);

impl ResourceSubscribers {
    /// The effects subscribed to the given resource.
    pub fn get(&self, resource: ComponentId) -> &[Entity] {
        self.0.get(&resource).map_or(&[], Vec::as_slice)
    }
}

/// The order an effect runs in, relative to other effects, within a flush.
///
/// Assigned from a monotonic counter when the effect is spawned, so ordering follows registration
/// order and is stable across runs. Without this the dirty queue would drain in whatever order
/// effects happened to be queued, which is incidental rather than defined.
#[derive(Component, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct EffectIndex(pub u64);

/// The queue of effects that may need to re-run, plus the tick of the last [`poll_sources`].
#[derive(Resource, Default)]
pub struct DirtyEffects {
    queue: Vec<Entity>,
    queued: HashSet<Entity>,
    last_poll: Tick,
    next_index: u64,
}

impl DirtyEffects {
    /// Queues `effect` to be re-checked and possibly re-run, if it isn't already queued.
    ///
    /// This only marks the effect as *maybe*-dirty. Whether it actually re-runs is decided by the
    /// pull check against its dependencies' ticks.
    pub fn push(&mut self, effect: Entity) {
        if self.queued.insert(effect) {
            self.queue.push(effect);
        }
    }

    /// Returns true if no effects are currently queued.
    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }

    fn take(&mut self) -> Vec<Entity> {
        self.queued.clear();
        core::mem::take(&mut self.queue)
    }

    /// Allocates the next [`EffectIndex`].
    fn next_index(&mut self) -> u64 {
        let index = self.next_index;
        self.next_index += 1;
        index
    }
}

/// The context passed to an effect body while it runs.
///
/// Use [`EffectContext::track_component`] or [`EffectContext::track`] to read reactively, and
/// [`EffectContext::world`] for arbitrary world access.
pub struct EffectContext<'w> {
    world: &'w mut World,
    effect: Entity,
    target: Option<Entity>,
    tracked: Vec<Source>,
}

impl EffectContext<'_> {
    /// The entity backing the currently running effect.
    pub fn effect(&self) -> Entity {
        self.effect
    }

    /// The entity this effect is attached to, if it was spawned with [`spawn_effect_for`].
    ///
    /// This is what [`Own`](source::Own) tracks against, so a reactive field can depend on a
    /// component of its own entity without any entity plumbing.
    pub fn target(&self) -> Option<Entity> {
        self.target
    }

    /// Mutable access to the [`World`].
    ///
    /// Reads performed through this world are *not* tracked.
    pub fn world(&mut self) -> &mut World {
        self.world
    }

    /// Reads component `C` from `entity` and records a dependency on it, so this effect re-runs
    /// whenever that component changes.
    ///
    /// Returns `None` if the entity or component is absent. The dependency is recorded either way,
    /// so an effect will re-run if a currently-missing component is later inserted.
    ///
    /// Note that change detection is per-component: a write to *any* field of `C` re-runs this
    /// effect. For a finer-grained source, use a [`Signal`].
    pub fn track_component<C: Component + Clone>(&mut self, entity: Entity) -> Option<C> {
        let component = self.world.register_component::<C>();
        self.record(Source::Component { entity, component });
        self.world.get::<C>(entity).cloned()
    }

    /// Reads `signal`'s current value and records a dependency on it.
    ///
    /// Returns `None` if the signal entity has been despawned.
    pub fn track<T: Clone + Send + Sync + 'static>(&mut self, signal: Signal<T>) -> Option<T> {
        let component = self.world.register_component::<SignalValue<T>>();
        self.record(Source::Component {
            entity: signal.entity,
            component,
        });
        self.world
            .get::<SignalValue<T>>(signal.entity)
            .map(|value| value.0.clone())
    }

    /// Reads resource `R` and records a dependency on it, so this effect re-runs whenever the
    /// resource changes.
    ///
    /// Returns `None` if the resource is absent. The dependency is recorded either way, so an
    /// effect will re-run if a currently-missing resource is later inserted.
    pub fn track_resource<R: Resource + Clone>(&mut self) -> Option<R> {
        let resource = self.world.register_component::<R>();
        self.record(Source::Resource { resource });
        self.world.get_resource::<R>().cloned()
    }

    /// Records a dependency on `source` without reading it.
    pub fn track_source(&mut self, source: Source) {
        self.record(source);
    }

    fn record(&mut self, source: Source) {
        if !self.tracked.contains(&source) {
            self.tracked.push(source);
        }
    }
}

/// Spawns a standalone effect entity and runs `func` once immediately to record its initial
/// dependencies.
///
/// The effect entity is never despawned automatically. To tie an effect's lifetime to another
/// entity, use [`spawn_effect_for`].
pub fn spawn_effect(
    world: &mut World,
    func: impl Fn(&mut EffectContext) + Send + Sync + 'static,
) -> Entity {
    spawn_effect_inner(world, None, (), func)
}

/// Spawns an effect entity related to `target` via [`EffectOf`], so it is despawned along with
/// `target`. Runs `func` once immediately.
pub fn spawn_effect_for(
    world: &mut World,
    target: Entity,
    func: impl Fn(&mut EffectContext) + Send + Sync + 'static,
) -> Entity {
    spawn_effect_inner(world, Some(target), (), func)
}

/// Spawns an effect entity, optionally related to `target`, with `extra` components inserted
/// **before** the first run. That matters for effects whose body reads state off their own entity
/// (see [`ProducerSystem`](field_effect::ProducerSystem)).
pub(crate) fn spawn_effect_inner(
    world: &mut World,
    target: Option<Entity>,
    extra: impl crate::bundle::Bundle,
    func: impl Fn(&mut EffectContext) + Send + Sync + 'static,
) -> Entity {
    let index = world.get_resource_or_init::<DirtyEffects>().next_index();
    let mut entity = world.spawn((
        Effect(Arc::new(func)),
        Dependencies::default(),
        EffectIndex(index),
        extra,
    ));
    if let Some(target) = target {
        entity.insert(EffectOf(target));
    }
    let effect = entity.id();
    run_effect(world, effect);
    effect
}

/// Runs `effect` only if one of its dependencies' `changed` ticks advanced past the tick it
/// observed on its last run. Returns whether the effect actually ran.
///
/// This is the "pull" half of the update flow.
fn maybe_run_effect(world: &mut World, effect: Entity) -> bool {
    if !dependencies_advanced(world, effect) {
        return false;
    }
    run_effect(world, effect);
    true
}

/// Returns whether any of `effect`'s dependencies changed since it last ran.
///
/// An effect with no recorded dependencies has never run (or tracked nothing), so it is always
/// considered dirty.
fn dependencies_advanced(world: &World, effect: Entity) -> bool {
    let Some(dependencies) = world.get::<Dependencies>(effect) else {
        return true;
    };
    if dependencies.0.is_empty() {
        return true;
    }
    let now = world.read_change_tick();
    dependencies.0.iter().any(|dependency| {
        source_changed_tick(world, dependency.source)
            .is_some_and(|changed| changed.is_newer_than(dependency.observed, now))
    })
}

/// Returns a source's current `changed` tick, if the source exists.
fn source_changed_tick(world: &World, source: Source) -> Option<Tick> {
    match source {
        Source::Component { entity, component } => world
            .get_entity(entity)
            .ok()
            .and_then(|entity| entity.get_change_ticks_by_id(component))
            .map(|ticks| ticks.changed),
        Source::Resource { resource } => world
            .get_resource_change_ticks_by_id(resource)
            .map(|ticks| ticks.changed),
    }
}

/// Runs a single effect unconditionally, then reconciles its dependency edges and captures the
/// ticks it observed.
fn run_effect(world: &mut World, effect: Entity) {
    let Some(Effect(func)) = world.get::<Effect>(effect).cloned() else {
        return;
    };

    // Give this run its own tick epoch, the same way the schedule executor does for each system.
    // `World::get_mut` reuses the current tick rather than advancing it, so without this every
    // write in a flush would be stamped identically and the pull check could not distinguish
    // "changed since I last ran" from "unchanged".
    world.increment_change_tick();

    let target = world.get::<EffectOf>(effect).map(|effect_of| effect_of.0);
    let mut context = EffectContext {
        world,
        effect,
        target,
        tracked: Vec::new(),
    };
    (func)(&mut context);
    let tracked = core::mem::take(&mut context.tracked);

    reconcile_dependencies(world, effect, tracked);

    // Close this run's tick epoch, so anything written afterwards is stamped strictly newer than
    // the ticks just observed. Without this, a source that was *absent* during the run records the
    // current tick, and a later insert stamped with that same tick would look unchanged.
    world.increment_change_tick();
}

/// Updates the subscription edges for `effect` to exactly `tracked`, and records the current
/// `changed` tick of each source.
///
/// Capturing the ticks *here* — after the body has run, and therefore after any writes it made —
/// is what prevents an effect from re-triggering itself.
fn reconcile_dependencies(world: &mut World, effect: Entity, tracked: Vec<Source>) {
    let previous: Vec<Source> = world
        .get::<Dependencies>(effect)
        .map(|deps| deps.0.iter().map(|dep| dep.source).collect())
        .unwrap_or_default();

    for source in &previous {
        if !tracked.contains(source) {
            unsubscribe(world, *source, effect);
        }
    }
    for source in &tracked {
        if !previous.contains(source) {
            subscribe(world, *source, effect);
        }
    }

    let now = world.change_tick();
    let dependencies: Vec<Dependency> = tracked
        .into_iter()
        .map(|source| {
            // A source that is absent right now is recorded at the current tick, so it counts as
            // changed once it is actually inserted.
            let observed = source_changed_tick(world, source).unwrap_or(now);
            Dependency { source, observed }
        })
        .collect();

    if let Some(mut slot) = world.get_mut::<Dependencies>(effect) {
        slot.0 = dependencies;
    }
}

fn subscribe(world: &mut World, source: Source, effect: Entity) {
    match source {
        Source::Component { entity, component } => {
            let Ok(mut entity) = world.get_entity_mut(entity) else {
                return;
            };
            if !entity.contains::<Subscribers>() {
                entity.insert(Subscribers::default());
            }
            if let Some(mut subscribers) = entity.get_mut::<Subscribers>() {
                let subs = subscribers.0.entry(component).or_default();
                if !subs.contains(&effect) {
                    subs.push(effect);
                }
            }
        }
        Source::Resource { resource } => {
            let mut subscribers = world.get_resource_or_init::<ResourceSubscribers>();
            let subs = subscribers.0.entry(resource).or_default();
            if !subs.contains(&effect) {
                subs.push(effect);
            }
        }
    }
}

fn unsubscribe(world: &mut World, source: Source, effect: Entity) {
    match source {
        Source::Component { entity, component } => {
            if let Some(mut subscribers) = world.get_mut::<Subscribers>(entity)
                && let Some(subs) = subscribers.0.get_mut(&component)
            {
                subs.retain(|subscriber| *subscriber != effect);
            }
        }
        Source::Resource { resource } => {
            if let Some(mut subscribers) = world.get_resource_mut::<ResourceSubscribers>()
                && let Some(subs) = subscribers.0.get_mut(&resource)
            {
                subs.retain(|subscriber| *subscriber != effect);
            }
        }
    }
}

/// Scans every tracked source and queues the subscribers of any whose `changed` tick advanced
/// since the last poll.
///
/// This is the "push" half of the update flow. It is deliberately conservative: it only marks
/// effects as maybe-dirty, and the per-effect pull check decides whether a re-run is actually
/// warranted. Cost is proportional to the number of *tracked* sources, not the size of the world.
///
/// If mutation observers land in `bevy_ecs`, this is the piece they would replace — the graph and
/// the pull check are unaffected.
pub fn poll_sources(world: &mut World) {
    let now = world.change_tick();
    let last_poll = world.get_resource_or_init::<DirtyEffects>().last_poll;

    let mut dirty: Vec<Entity> = Vec::new();
    // PERF: this builds a fresh QueryState per poll. A cached SystemState would avoid that.
    let mut sources = world.query::<(Entity, &Subscribers)>();
    {
        let world: &World = world;
        for (entity, subscribers) in sources.iter(world) {
            let Ok(entity_ref) = world.get_entity(entity) else {
                continue;
            };
            for (component, subs) in subscribers.0.iter() {
                if subs.is_empty() {
                    continue;
                }
                if entity_ref
                    .get_change_ticks_by_id(*component)
                    .is_some_and(|ticks| ticks.changed.is_newer_than(last_poll, now))
                {
                    dirty.extend(subs.iter().copied());
                }
            }
        }
    }

    // Resource sources, whose subscriber lists live in a resource rather than on an entity.
    if let Some(subscribers) = world.remove_resource::<ResourceSubscribers>() {
        for (resource, subs) in subscribers.0.iter() {
            if !subs.is_empty()
                && world
                    .get_resource_change_ticks_by_id(*resource)
                    .is_some_and(|ticks| ticks.changed.is_newer_than(last_poll, now))
            {
                dirty.extend(subs.iter().copied());
            }
        }
        world.insert_resource(subscribers);
    }

    let mut queue = world.get_resource_or_init::<DirtyEffects>();
    for effect in dirty {
        queue.push(effect);
    }
    queue.last_poll = now;

    // Ensure any write made *after* this poll is stamped strictly newer than `last_poll`.
    // `World::get_mut` reuses the current tick, so without this a write occurring between two
    // polls at the same tick would be invisible to the next poll.
    world.increment_change_tick();
}

impl World {
    /// Spawns a new signal entity holding `value` and returns a handle to it.
    pub fn spawn_signal<T: Send + Sync + 'static>(&mut self, value: T) -> Signal<T> {
        let entity = self.spawn(SignalValue(value)).id();
        Signal::from_entity(entity)
    }

    /// Reads a signal's current value _without_ recording a dependency.
    ///
    /// Inside an effect, prefer [`EffectContext::track`], which subscribes the effect to the
    /// signal. Returns `None` if the signal entity has been despawned.
    pub fn signal<T: Send + Sync + 'static>(&self, signal: Signal<T>) -> Option<&T> {
        self.get::<SignalValue<T>>(signal.entity()).map(|v| &v.0)
    }

    /// Mutably accesses a signal's value.
    ///
    /// The signal is marked changed only if the returned guard is actually dereferenced mutably,
    /// so a read-only inspection does not wake subscribers. Subscribed effects do not run until
    /// [`World::run_effects`] is called. Returns `None` if the signal entity has been despawned.
    pub fn signal_mut<T: Send + Sync + 'static>(
        &mut self,
        signal: Signal<T>,
    ) -> Option<Mut<'_, T>> {
        self.get_mut::<SignalValue<T>>(signal.entity())
            .map(|value| value.map_unchanged(|v| &mut v.0))
    }

    /// Polls tracked sources and runs every effect whose dependencies actually changed, repeating
    /// until nothing further is dirty.
    ///
    /// Effects that write sources other effects depend on cascade within this same call. If things
    /// have not settled after [`MAX_EFFECT_ITERATIONS`] passes, this gives up and logs an error,
    /// which indicates genuine oscillation in the effect graph.
    pub fn run_effects(&mut self) {
        for _ in 0..MAX_EFFECT_ITERATIONS {
            poll_sources(self);

            let Some(mut dirty) = self.get_resource_mut::<DirtyEffects>() else {
                return;
            };
            if dirty.is_empty() {
                return;
            }
            let mut effects = dirty.take();
            // Run in registration order so a flush is deterministic regardless of the order in
            // which sources happened to be marked dirty.
            effects.sort_by_key(|effect| self.get::<EffectIndex>(*effect).map(|i| i.0));
            for effect in effects {
                maybe_run_effect(self, effect);
            }
        }

        if self
            .get_resource::<DirtyEffects>()
            .is_some_and(|dirty| !dirty.is_empty())
        {
            log::error!(
                "Effects did not settle after {MAX_EFFECT_ITERATIONS} iterations. This usually \
                 means effects in the graph keep changing each other's values. The remaining \
                 effects will run on the next flush."
            );
        }
    }
}

/// A system that polls sources and runs every dirty effect. See [`World::run_effects`].
pub fn run_effects(world: &mut World) {
    world.run_effects();
}

/// A boxed, one-shot registration that spawns an effect for a target entity.
///
/// See [`QueuedEffects`] and [`TemplateContext::queue_effect`](crate::template::TemplateContext::queue_effect).
pub type BoxedEffectRegistration = Box<dyn FnOnce(&mut World, Entity) + Send + Sync>;

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Component, Default, Clone)]
    struct Value(u32);

    #[derive(Component, Default, Clone)]
    struct Health {
        current: u32,
        max: u32,
    }

    fn write_to(target: Entity, signal: Signal<u32>) -> impl Fn(&mut EffectContext) {
        move |ctx| {
            let value = ctx.track(signal).unwrap_or_default();
            if let Some(mut slot) = ctx.world().get_mut::<Value>(target) {
                slot.0 = value;
            }
        }
    }

    #[test]
    fn effect_runs_on_spawn_and_on_set() {
        let mut world = World::new();
        let signal = world.spawn_signal(1u32);
        let target = world.spawn(Value::default()).id();

        spawn_effect(&mut world, write_to(target, signal));
        assert_eq!(world.get::<Value>(target).unwrap().0, 1);

        *world.signal_mut(signal).unwrap() = 7;
        assert_eq!(
            world.get::<Value>(target).unwrap().0,
            1,
            "setting alone does not run effects"
        );

        world.run_effects();
        assert_eq!(world.get::<Value>(target).unwrap().0, 7);
    }

    #[test]
    fn plain_component_mutation_is_a_reactive_source() {
        let mut world = World::new();
        let source = world
            .spawn(Health {
                current: 3,
                max: 10,
            })
            .id();
        let target = world.spawn(Value::default()).id();

        spawn_effect(&mut world, move |ctx| {
            let current = ctx
                .track_component::<Health>(source)
                .map_or(0, |h| h.current);
            if let Some(mut slot) = ctx.world().get_mut::<Value>(target) {
                slot.0 = current;
            }
        });
        assert_eq!(world.get::<Value>(target).unwrap().0, 3);

        // Mutated the way any ordinary system would: no signal, no handle, no commands.
        world.get_mut::<Health>(source).unwrap().current = 9;
        world.run_effects();
        assert_eq!(world.get::<Value>(target).unwrap().0, 9);
    }

    #[test]
    fn untracked_source_does_not_re_run_effect() {
        let mut world = World::new();
        let tracked = world.spawn_signal(1u32);
        let untracked = world.spawn_signal(100u32);
        let target = world.spawn(Value::default()).id();

        spawn_effect(&mut world, write_to(target, tracked));

        *world.signal_mut(untracked).unwrap() = 500;
        world.run_effects();
        assert_eq!(world.get::<Value>(target).unwrap().0, 1);
    }

    #[test]
    fn inspecting_a_signal_without_writing_does_not_wake_subscribers() {
        let mut world = World::new();
        let signal = world.spawn_signal(1u32);
        let runs = world.spawn(Value::default()).id();

        spawn_effect(&mut world, move |ctx| {
            ctx.track(signal);
            if let Some(mut slot) = ctx.world().get_mut::<Value>(runs) {
                slot.0 += 1;
            }
        });
        assert_eq!(world.get::<Value>(runs).unwrap().0, 1);

        // Take the guard and only read through it. `Mut` marks changed on deref-*mut*, so this
        // must not wake the effect.
        let observed = *world.signal_mut(signal).unwrap();
        assert_eq!(observed, 1);

        world.run_effects();
        assert_eq!(
            world.get::<Value>(runs).unwrap().0,
            1,
            "a read-only `signal_mut` guard must not mark the signal changed"
        );

        // ...but writing through it does.
        *world.signal_mut(signal).unwrap() = 2;
        world.run_effects();
        assert_eq!(world.get::<Value>(runs).unwrap().0, 2);
    }

    #[test]
    fn dependencies_are_re_recorded_each_run() {
        let mut world = World::new();
        let toggle = world.spawn_signal(false);
        let a = world.spawn_signal(1u32);
        let b = world.spawn_signal(2u32);
        let target = world.spawn(Value::default()).id();

        spawn_effect(&mut world, move |ctx| {
            let use_b = ctx.track(toggle).unwrap_or_default();
            let value = if use_b {
                ctx.track(b).unwrap_or_default()
            } else {
                ctx.track(a).unwrap_or_default()
            };
            if let Some(mut slot) = ctx.world().get_mut::<Value>(target) {
                slot.0 = value;
            }
        });
        assert_eq!(world.get::<Value>(target).unwrap().0, 1);

        *world.signal_mut(b).unwrap() = 20;
        world.run_effects();
        assert_eq!(
            world.get::<Value>(target).unwrap().0,
            1,
            "`b` is not tracked yet"
        );

        *world.signal_mut(toggle).unwrap() = true;
        world.run_effects();
        assert_eq!(world.get::<Value>(target).unwrap().0, 20);

        *world.signal_mut(a).unwrap() = 10;
        world.run_effects();
        assert_eq!(
            world.get::<Value>(target).unwrap().0,
            20,
            "`a` is no longer tracked"
        );

        *world.signal_mut(b).unwrap() = 30;
        world.run_effects();
        assert_eq!(world.get::<Value>(target).unwrap().0, 30);

        assert!(
            world
                .get::<Subscribers>(a.entity())
                .is_none_or(Subscribers::is_empty),
            "`a` should have been unsubscribed once it stopped being read"
        );
    }

    #[test]
    fn effects_cascade_within_one_flush() {
        let mut world = World::new();
        let source = world.spawn_signal(1u32);
        let doubled = world.spawn_signal(0u32);
        let target = world.spawn(Value::default()).id();

        spawn_effect(&mut world, move |ctx| {
            let value = ctx.track(source).unwrap_or_default() * 2;
            *ctx.world().signal_mut(doubled).unwrap() = value;
        });
        spawn_effect(&mut world, write_to(target, doubled));

        *world.signal_mut(source).unwrap() = 5;
        world.run_effects();
        assert_eq!(world.get::<Value>(target).unwrap().0, 10);
    }

    /// The cycle case: an effect that both reads and writes the same component. The post-run tick
    /// capture must make this settle rather than spin.
    #[test]
    fn effect_writing_a_component_it_reads_does_not_re_trigger_itself() {
        let mut world = World::new();
        let source = world
            .spawn(Health {
                current: 0,
                max: 10,
            })
            .id();
        let runs = world.spawn(Value::default()).id();

        spawn_effect(&mut world, move |ctx| {
            // Read and write the very same component.
            let max = ctx.track_component::<Health>(source).map_or(0, |h| h.max);
            if let Some(mut health) = ctx.world().get_mut::<Health>(source) {
                health.current = max;
            }
            if let Some(mut slot) = ctx.world().get_mut::<Value>(runs) {
                slot.0 += 1;
            }
        });
        assert_eq!(world.get::<Value>(runs).unwrap().0, 1);
        assert_eq!(world.get::<Health>(source).unwrap().current, 10);

        // A flush with no external change must not re-run it at all.
        world.run_effects();
        assert_eq!(
            world.get::<Value>(runs).unwrap().0,
            1,
            "the effect's own write must not re-trigger it"
        );

        // An external change still re-runs it exactly once.
        world.get_mut::<Health>(source).unwrap().max = 20;
        world.run_effects();
        assert_eq!(world.get::<Value>(runs).unwrap().0, 2);
        assert_eq!(world.get::<Health>(source).unwrap().current, 20);

        world.run_effects();
        assert_eq!(
            world.get::<Value>(runs).unwrap().0,
            2,
            "should have settled again"
        );
    }

    /// A chain through a second effect must settle, and derivations must still propagate — which
    /// suppressing change detection on effect writes would have broken.
    #[test]
    fn chained_effects_settle_and_still_propagate() {
        let mut world = World::new();
        let a = world.spawn(Value(1)).id();
        let b = world.spawn(Value(0)).id();
        let runs = world.spawn(Value::default()).id();

        // b = a * 2
        spawn_effect(&mut world, move |ctx| {
            let value = ctx.track_component::<Value>(a).map_or(0, |v| v.0);
            if let Some(mut slot) = ctx.world().get_mut::<Value>(b) {
                slot.0 = value * 2;
            }
            if let Some(mut slot) = ctx.world().get_mut::<Value>(runs) {
                slot.0 += 1;
            }
        });
        assert_eq!(world.get::<Value>(b).unwrap().0, 2);

        world.get_mut::<Value>(a).unwrap().0 = 5;
        world.run_effects();
        assert_eq!(
            world.get::<Value>(b).unwrap().0,
            10,
            "an effect's write is visible to downstream effects"
        );
        assert_eq!(world.get::<Value>(runs).unwrap().0, 2);

        world.run_effects();
        assert_eq!(
            world.get::<Value>(runs).unwrap().0,
            2,
            "no spurious re-runs once settled"
        );
    }

    #[test]
    fn no_op_flush_runs_nothing() {
        let mut world = World::new();
        let signal = world.spawn_signal(1u32);
        let runs = world.spawn(Value::default()).id();

        spawn_effect(&mut world, move |ctx| {
            ctx.track(signal);
            if let Some(mut slot) = ctx.world().get_mut::<Value>(runs) {
                slot.0 += 1;
            }
        });
        assert_eq!(world.get::<Value>(runs).unwrap().0, 1);

        for _ in 0..5 {
            world.run_effects();
        }
        assert_eq!(
            world.get::<Value>(runs).unwrap().0,
            1,
            "flushing without any change must not re-run effects"
        );
    }

    #[test]
    fn despawning_target_despawns_effect_and_unsubscribes() {
        let mut world = World::new();
        let signal = world.spawn_signal(1u32);
        let target = world.spawn(Value::default()).id();

        let effect = spawn_effect_for(&mut world, target, write_to(target, signal));
        assert!(!world
            .get::<Subscribers>(signal.entity())
            .unwrap()
            .is_empty());

        world.entity_mut(target).despawn();

        assert!(world.get_entity(effect).is_err(), "effect should despawn");
        assert!(
            world
                .get::<Subscribers>(signal.entity())
                .unwrap()
                .is_empty(),
            "despawned effect should be unsubscribed"
        );

        *world.signal_mut(signal).unwrap() = 2;
        world.run_effects();
    }

    #[test]
    fn effects_are_deduplicated_within_a_flush() {
        let mut world = World::new();
        let a = world.spawn_signal(1u32);
        let b = world.spawn_signal(1u32);
        let runs = world.spawn(Value::default()).id();

        spawn_effect(&mut world, move |ctx| {
            ctx.track(a);
            ctx.track(b);
            if let Some(mut slot) = ctx.world().get_mut::<Value>(runs) {
                slot.0 += 1;
            }
        });
        assert_eq!(world.get::<Value>(runs).unwrap().0, 1);

        *world.signal_mut(a).unwrap() = 2;
        *world.signal_mut(b).unwrap() = 2;
        world.run_effects();

        assert_eq!(
            world.get::<Value>(runs).unwrap().0,
            2,
            "two sets before one flush should produce a single re-run"
        );
    }

    #[derive(Resource, Default)]
    struct RunOrder(Vec<u64>);

    /// Effects run in registration order, not in whatever order they were marked dirty.
    #[test]
    fn effects_run_in_registration_order() {
        let mut world = World::new();
        world.init_resource::<RunOrder>();
        let signal = world.spawn_signal(0u32);

        let record = |tag: u64| {
            move |ctx: &mut EffectContext| {
                ctx.track(signal);
                ctx.world().resource_mut::<RunOrder>().0.push(tag);
            }
        };
        let first = spawn_effect(&mut world, record(1));
        let second = spawn_effect(&mut world, record(2));
        assert!(
            world.get::<EffectIndex>(first).unwrap() < world.get::<EffectIndex>(second).unwrap()
        );

        world.resource_mut::<RunOrder>().0.clear();

        // Queue them in the *reverse* of registration order before flushing.
        *world.signal_mut(signal).unwrap() = 1;
        let mut dirty = world.get_resource_or_init::<DirtyEffects>();
        dirty.push(second);
        dirty.push(first);
        world.run_effects();

        assert_eq!(
            world.resource::<RunOrder>().0,
            alloc::vec![1, 2],
            "the flush must sort by EffectIndex, not by queue order"
        );
    }

    /// A dependency is recorded even while its source is absent, so inserting the source later
    /// wakes the effect. This is easy to get wrong: an absent source records the current tick, so
    /// the run's trailing tick increment is what keeps a later insert strictly newer.
    #[test]
    fn a_source_inserted_later_wakes_the_effect() {
        let mut world = World::new();
        let source = world.spawn_empty().id();
        let target = world.spawn(Value::default()).id();

        spawn_effect(&mut world, move |ctx| {
            let current = ctx
                .track_component::<Health>(source)
                .map_or(0, |h| h.current);
            if let Some(mut slot) = ctx.world().get_mut::<Value>(target) {
                slot.0 = current;
            }
        });
        assert_eq!(world.get::<Value>(target).unwrap().0, 0);

        world.entity_mut(source).insert(Health {
            current: 5,
            max: 10,
        });
        world.run_effects();
        assert_eq!(world.get::<Value>(target).unwrap().0, 5);
    }

    #[derive(Resource, Default, Clone)]
    struct Scale(u32);

    #[test]
    fn resources_are_reactive_sources() {
        let mut world = World::new();
        world.insert_resource(Scale(2));
        let target = world.spawn(Value::default()).id();

        spawn_effect(&mut world, move |ctx| {
            let scale = ctx.track_resource::<Scale>().unwrap_or_default();
            if let Some(mut slot) = ctx.world().get_mut::<Value>(target) {
                slot.0 = scale.0 * 10;
            }
        });
        assert_eq!(world.get::<Value>(target).unwrap().0, 20);

        world.resource_mut::<Scale>().0 = 7;
        world.run_effects();
        assert_eq!(world.get::<Value>(target).unwrap().0, 70);

        // An unrelated flush must not re-run it.
        world.run_effects();
        assert_eq!(world.get::<Value>(target).unwrap().0, 70);
    }
}
