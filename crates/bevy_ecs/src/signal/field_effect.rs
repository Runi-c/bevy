//! The "surgical update" primitive: a [`Template`] that writes a _single field_ of a single
//! component in response to its [`ReactiveSources`] changing.

use super::{source::ReactiveSources, spawn_effect_for, BoxedEffectRegistration};
use crate::{
    component::{Component, Mutable},
    entity::Entity,
    error::Result,
    template::{Template, TemplateContext},
    world::World,
};
use alloc::{boxed::Box, vec::Vec};
use core::marker::PhantomData;

/// Effect registrations accumulated while building templates for one entity.
///
/// Effects cannot be spawned _during_ template building, because scene application batches every
/// component into a single [`BundleWriter`] write. Registering an effect would need world access
/// mid-flight, and the component the effect writes to does not exist on the entity yet. So
/// registrations are queued here and flushed by the caller once the entity is fully written.
///
/// [`BundleWriter`]: crate::bundle::BundleWriter
#[derive(Default)]
pub struct QueuedEffects(Vec<BoxedEffectRegistration>);

impl QueuedEffects {
    /// Returns true if no effects are queued.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Queues an effect registration, to be run with the target entity once it is fully written.
    pub fn push(&mut self, registration: BoxedEffectRegistration) {
        self.0.push(registration);
    }

    /// Takes the queued registrations, leaving this empty.
    pub fn take(&mut self) -> Vec<BoxedEffectRegistration> {
        core::mem::take(&mut self.0)
    }

    /// Runs every queued registration against `target` and clears the queue.
    pub fn flush(&mut self, world: &mut World, target: Entity) {
        for registration in self.take() {
            registration(world, target);
        }
    }
}

/// A [`Template`] that reactively writes one field of the component `C`.
///
/// It holds three parts:
/// - `sources`: what the field depends on — a [`ReactiveSource`], or a tuple of them.
/// - `producer`: a **pure** function from the sources' read values to the field's value. It gets no
///   world or effect access — everything it needs is already in the read values, which keeps the
///   field's dependencies exactly equal to its declared sources.
/// - `setter`: writes that value into a live `C`.
///
/// Building this template resolves the sources' build-time data (see [`ReactiveSource::resolve`])
/// and queues an effect registration on the [`TemplateContext`]. Once the entity has been fully
/// written, the effect is spawned (related to the entity via [`EffectOf`], so it despawns with it)
/// and run once, writing the initial value. From then on it re-runs whenever a tracked source
/// changes, and each run touches only `C` on that one entity — no archetype move, and no rebuild
/// of the rest of the scene.
///
/// `C` must already be present on the entity: the field write is a mutation, not an insert. Scene
/// code is responsible for also inserting `C`'s ordinary template.
///
/// [`ReactiveSource`]: super::source::ReactiveSource
/// [`ReactiveSource::resolve`]: super::source::ReactiveSource::resolve
/// [`EffectOf`]: super::EffectOf
///
/// ```
/// # use bevy_ecs::prelude::*;
/// # use bevy_ecs::signal::{FieldEffect, Signal};
/// # use bevy_ecs::template::Template;
/// #[derive(Component, Default, Clone)]
/// struct Health { current: u32, max: u32 }
///
/// # let mut world = World::new();
/// let hp = world.spawn_signal(50u32);
/// // Reactively drives `Health::current`, leaving `Health::max` alone.
/// let effect = FieldEffect::new(
///     hp,
///     |hp: u32| hp * 2,
///     |health: &mut Health, value| health.current = value,
/// );
///
/// // Scene application does this for you; done by hand here.
/// let target = world.spawn(Health { current: 0, max: 100 }).id();
/// let mut queued = world
///     .entity_mut(target)
///     .template_context(|ctx| {
///         effect.build_template(ctx)?;
///         Ok(ctx.take_effects())
///     })
///     .unwrap();
/// queued.flush(&mut world, target);
///
/// assert_eq!(world.get::<Health>(target).unwrap().current, 100);
/// assert_eq!(world.get::<Health>(target).unwrap().max, 100);
/// ```
pub struct FieldEffect<C, T, Src, P, S> {
    sources: Src,
    producer: P,
    setter: S,
    marker: PhantomData<fn() -> (C, T)>,
}

impl<C, T, Src, P, S> FieldEffect<C, T, Src, P, S> {
    /// Creates a new [`FieldEffect`] from its sources, a pure producer, and a field setter.
    pub fn new(sources: Src, producer: P, setter: S) -> Self {
        Self {
            sources,
            producer,
            setter,
            marker: PhantomData,
        }
    }
}

impl<C, T, Src, P, S> Template for FieldEffect<C, T, Src, P, S>
where
    C: Component<Mutability = Mutable>,
    T: Send + Sync + 'static,
    Src: ReactiveSources,
    P: Fn(Src::Output) -> T + Clone + Send + Sync + 'static,
    S: Fn(&mut C, T) + Clone + Send + Sync + 'static,
{
    // `()` is a `Bundle`, which lets this be applied as a bundle template. Nothing is inserted;
    // the real work happens when the queued registration is flushed.
    type Output = ();

    fn build_template(&self, context: &mut TemplateContext) -> Result<Self::Output> {
        // Build-time phase: this is the only point where `#Name` references can be turned into
        // real entities. The result is baked into the effect closure.
        let resolved = self.sources.resolve_all(context)?;

        let sources = self.sources.clone();
        let producer = self.producer.clone();
        let setter = self.setter.clone();
        context.queue_effect(Box::new(move |world: &mut World, target: Entity| {
            spawn_effect_for(world, target, move |ctx| {
                let values = sources.read_all(&resolved, ctx);
                let value = (producer)(values);
                if let Some(mut component) = ctx.world().get_mut::<C>(target) {
                    (setter)(&mut component, value);
                }
            });
        }));
        Ok(())
    }

    fn clone_template(&self) -> Self {
        Self {
            sources: self.sources.clone(),
            producer: self.producer.clone(),
            setter: self.setter.clone(),
            marker: PhantomData,
        }
    }
}

/// The value a [`DedupedFieldEffect`] produced on its last run, cached on the effect entity.
#[derive(Component)]
pub struct LastValue<T: Send + Sync + 'static>(pub T);

/// A [`FieldEffect`] that skips the component write when the produced value is unchanged.
///
/// This is the equality cutoff that keeps deep reactive graphs cheap. Change ticks bump on any
/// `deref_mut`, even when the written value is identical, so without dedup an effect that
/// recomputes to the same value still marks its target component changed and needlessly re-runs
/// everything downstream of it.
///
/// The last produced value is cached on the effect entity in a [`LastValue<T>`], so no getter for
/// the field is required — dedup happens on the producer's output, before the write.
///
/// Prefer this over [`FieldEffect`] whenever `T: PartialEq`.
pub struct DedupedFieldEffect<C, T, Src, P, S>(FieldEffect<C, T, Src, P, S>);

impl<C, T, Src, P, S> DedupedFieldEffect<C, T, Src, P, S> {
    /// Creates a new [`DedupedFieldEffect`]. See [`FieldEffect::new`].
    pub fn new(sources: Src, producer: P, setter: S) -> Self {
        Self(FieldEffect::new(sources, producer, setter))
    }
}

impl<C, T, Src, P, S> Template for DedupedFieldEffect<C, T, Src, P, S>
where
    C: Component<Mutability = Mutable>,
    T: PartialEq + Clone + Send + Sync + 'static,
    Src: ReactiveSources,
    P: Fn(Src::Output) -> T + Clone + Send + Sync + 'static,
    S: Fn(&mut C, T) + Clone + Send + Sync + 'static,
{
    type Output = ();

    fn build_template(&self, context: &mut TemplateContext) -> Result<Self::Output> {
        let resolved = self.0.sources.resolve_all(context)?;

        let sources = self.0.sources.clone();
        let producer = self.0.producer.clone();
        let setter = self.0.setter.clone();
        context.queue_effect(Box::new(move |world: &mut World, target: Entity| {
            spawn_effect_for(world, target, move |ctx| {
                let values = sources.read_all(&resolved, ctx);
                let value = (producer)(values);
                let effect = ctx.effect();

                // Unchanged output: skip the write entirely, leaving the component's change tick
                // untouched so nothing downstream is woken.
                if ctx
                    .world()
                    .get::<LastValue<T>>(effect)
                    .is_some_and(|last| last.0 == value)
                {
                    return;
                }
                ctx.world()
                    .entity_mut(effect)
                    .insert(LastValue(value.clone()));

                if let Some(mut component) = ctx.world().get_mut::<C>(target) {
                    (setter)(&mut component, value);
                }
            });
        }));
        Ok(())
    }

    fn clone_template(&self) -> Self {
        Self(Template::clone_template(&self.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::signal::source::Own;

    #[derive(Component, Default, Clone, PartialEq)]
    struct Health {
        current: u32,
        max: u32,
    }

    #[derive(Component, Default, Clone)]
    struct Label(u32);

    /// Builds `effect` against `target` the way scene application does, then flushes the queued
    /// registration.
    fn apply(world: &mut World, target: Entity, effect: &impl Template<Output = ()>) {
        let mut entity = world.entity_mut(target);
        let mut queued = entity
            .template_context(|context| {
                effect.build_template(context)?;
                Ok(context.take_effects())
            })
            .unwrap();
        queued.flush(world, target);
    }

    #[test]
    fn field_effect_writes_only_its_own_field() {
        let mut world = World::new();
        let hp = world.spawn_signal(10u32);

        let effect: FieldEffect<Health, u32, _, _, _> = FieldEffect::new(
            hp,
            |hp: u32| hp,
            |health: &mut Health, value| health.current = value,
        );

        let target = world
            .spawn(Health {
                current: 0,
                max: 100,
            })
            .id();
        apply(&mut world, target, &effect);

        let health = world.get::<Health>(target).unwrap();
        assert_eq!(health.current, 10, "initial value should be written");
        assert_eq!(health.max, 100, "unrelated fields must be untouched");

        *world.signal_mut(hp).unwrap() = 42;
        world.run_effects();

        let health = world.get::<Health>(target).unwrap();
        assert_eq!(health.current, 42);
        assert_eq!(health.max, 100);
    }

    /// A deduped effect must not bump its target component's change tick when the produced value
    /// is unchanged — that is what stops redundant propagation downstream.
    #[test]
    fn deduped_field_effect_skips_redundant_writes() {
        let mut world = World::new();
        let hp = world.spawn_signal(10u32);

        // Deliberately lossy: distinct signal values can map to the same output.
        let effect: DedupedFieldEffect<Health, u32, _, _, _> = DedupedFieldEffect::new(
            hp,
            |hp: u32| hp.min(10),
            |health: &mut Health, value| health.current = value,
        );

        let target = world
            .spawn(Health {
                current: 0,
                max: 100,
            })
            .id();
        apply(&mut world, target, &effect);
        assert_eq!(world.get::<Health>(target).unwrap().current, 10);

        let tick_before = world
            .entity(target)
            .get_change_ticks::<Health>()
            .unwrap()
            .changed;

        *world.signal_mut(hp).unwrap() = 999;
        world.run_effects();

        assert_eq!(world.get::<Health>(target).unwrap().current, 10);
        assert_eq!(
            world
                .entity(target)
                .get_change_ticks::<Health>()
                .unwrap()
                .changed,
            tick_before,
            "an unchanged value must not mark the target component changed"
        );

        *world.signal_mut(hp).unwrap() = 4;
        world.run_effects();
        assert_eq!(world.get::<Health>(target).unwrap().current, 4);
    }

    /// `Own<C>` tracks a component on the effect's own entity — no entity plumbing at all.
    #[test]
    fn own_source_tracks_the_effects_own_entity() {
        let mut world = World::new();

        let effect: FieldEffect<Label, u32, _, _, _> = FieldEffect::new(
            (Own::<Health>::default(),),
            |(health,): (Health,)| health.current * 2,
            |label: &mut Label, value| label.0 = value,
        );

        let target = world
            .spawn((
                Health {
                    current: 3,
                    max: 10,
                },
                Label::default(),
            ))
            .id();
        apply(&mut world, target, &effect);

        assert_eq!(world.get::<Label>(target).unwrap().0, 6);

        // Ordinary mutation of the same entity's component drives the update.
        world.get_mut::<Health>(target).unwrap().current = 8;
        world.run_effects();
        assert_eq!(world.get::<Label>(target).unwrap().0, 16);
    }

    /// Multiple sources of different kinds combine into one pure producer.
    #[test]
    fn a_bare_source_needs_no_tuple() {
        let mut world = World::new();

        // Both a lone source and a 1-tuple of it are valid `ReactiveSources`, and the producer's
        // parameter shape follows: a bare value in the first case, a 1-tuple in the second.
        let bare: FieldEffect<Label, u32, _, _, _> = FieldEffect::new(
            Own::<Health>::default(),
            |health: Health| health.current,
            |label: &mut Label, value| label.0 = value,
        );
        let tupled: FieldEffect<Label, u32, _, _, _> = FieldEffect::new(
            (Own::<Health>::default(),),
            |(health,): (Health,)| health.current,
            |label: &mut Label, value| label.0 = value,
        );

        let target = world
            .spawn((
                Health {
                    current: 3,
                    max: 10,
                },
                Label::default(),
            ))
            .id();
        apply(&mut world, target, &bare);
        assert_eq!(world.get::<Label>(target).unwrap().0, 3);

        world.get_mut::<Health>(target).unwrap().current = 8;
        world.run_effects();
        assert_eq!(
            world.get::<Label>(target).unwrap().0,
            8,
            "bare source reacts"
        );

        let target = world
            .spawn((
                Health {
                    current: 3,
                    max: 10,
                },
                Label::default(),
            ))
            .id();
        apply(&mut world, target, &tupled);
        assert_eq!(world.get::<Label>(target).unwrap().0, 3);
    }

    #[test]
    fn multiple_sources_combine() {
        let mut world = World::new();
        let bonus = world.spawn_signal(100u32);

        let effect: FieldEffect<Label, u32, _, _, _> = FieldEffect::new(
            (Own::<Health>::default(), bonus),
            |(health, bonus): (Health, u32)| health.current + bonus,
            |label: &mut Label, value| label.0 = value,
        );

        let target = world
            .spawn((
                Health {
                    current: 1,
                    max: 10,
                },
                Label::default(),
            ))
            .id();
        apply(&mut world, target, &effect);
        assert_eq!(world.get::<Label>(target).unwrap().0, 101);

        // Either source re-runs the effect.
        world.get_mut::<Health>(target).unwrap().current = 5;
        world.run_effects();
        assert_eq!(world.get::<Label>(target).unwrap().0, 105);

        *world.signal_mut(bonus).unwrap() = 200;
        world.run_effects();
        assert_eq!(world.get::<Label>(target).unwrap().0, 205);
    }
}
