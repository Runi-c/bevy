//! Scene-level glue for fine-grained reactivity.
//!
//! This exposes [`bevy_ecs::signal`]'s [`FieldEffect`] as a [`Scene`], so a reactive field can be
//! composed alongside ordinary patches.
//!
//! [`reactive_field`] is what the `bsn!` macro's `$(...)` syntax expands to. Using it directly is
//! useful when you need a field path the macro cannot express, or a custom setter. See the
//! ["Reactive Fields"](crate#reactive-fields-experimental) section of the crate docs.

use crate::{ResolveContext, ResolvedScene, Scene, SceneFunction};
use bevy_ecs::{
    component::{Component, Mutable},
    signal::{source::ReactiveSources, DedupedFieldEffect, FieldEffect},
    template::FromTemplate,
};

/// Returns a [`Scene`] that reactively drives a single field of the component `C`.
///
/// `sources` is what the field depends on; `producer` is a **pure** function from the sources'
/// read values to the field's value; `setter` writes that value into a live `C`.
///
/// This does two things to the resolved scene:
/// 1. Ensures `C`'s ordinary template is present, so `C` exists on the entity (with its other
///    fields patched as usual) after the scene's single bundle write.
/// 2. Pushes a [`FieldEffect`], which registers an effect once the entity is written.
///
/// Updates are surgical: when a tracked signal changes, only this field of this component on this
/// entity is written. Nothing is re-resolved, no other component is touched, and there is no
/// archetype move.
///
/// The effect's lifetime is tied to the spawned entity, so it is despawned along with it.
///
/// ```
/// # use bevy_app::App;
/// # use bevy_scene::{prelude::*, reactive_field, ScenePlugin};
/// # use bevy_ecs::{prelude::*, signal::Signal};
/// # use bevy_asset::AssetPlugin;
/// # use bevy_app::TaskPoolPlugin;
/// # let mut app = App::new();
/// # app.add_plugins((TaskPoolPlugin::default(), AssetPlugin::default(), ScenePlugin));
/// # let world = app.world_mut();
/// #[derive(Component, Default, Clone)]
/// struct Health { current: u32, max: u32 }
///
/// let hp = world.spawn_signal(30u32);
///
/// world.spawn_scene(bsn! {
///     Health { max: 100 }
///     {reactive_field(
///         hp,
///         |hp: u32| hp,
///         |health: &mut Health, value| health.current = value,
///     )}
/// }).unwrap();
/// ```
pub fn reactive_field<C, T, Src, P, S>(sources: Src, producer: P, setter: S) -> impl Scene
where
    C: Component<Mutability = Mutable> + FromTemplate,
    C::Template: Default + Send + Sync + 'static,
    T: Send + Sync + 'static,
    Src: ReactiveSources,
    P: Fn(Src::Output) -> T + Clone + Send + Sync + 'static,
    S: Fn(&mut C, T) + Clone + Send + Sync + 'static,
{
    SceneFunction(
        move |context: &mut ResolveContext, scene: &mut ResolvedScene| {
            // Make sure `C` itself is part of the scene; the effect mutates it, it does not insert it.
            let _ = scene.get_or_insert_template::<C::Template>(context);
            scene.push_bundle_template(FieldEffect::<C, T, Src, P, S>::new(
                sources, producer, setter,
            ));
        },
    )
}

/// Like [`reactive_field`], but skips the write when the produced value is unchanged.
///
/// Prefer this whenever `T: PartialEq`. Because change ticks bump on any write, a redundant write
/// marks the target component changed and needlessly wakes everything tracking it; deduping cuts
/// that propagation off at the source. See [`DedupedFieldEffect`].
pub fn reactive_field_deduped<C, T, Src, P, S>(sources: Src, producer: P, setter: S) -> impl Scene
where
    C: Component<Mutability = Mutable> + FromTemplate,
    C::Template: Default + Send + Sync + 'static,
    T: PartialEq + Clone + Send + Sync + 'static,
    Src: ReactiveSources,
    P: Fn(Src::Output) -> T + Clone + Send + Sync + 'static,
    S: Fn(&mut C, T) + Clone + Send + Sync + 'static,
{
    SceneFunction(
        move |context: &mut ResolveContext, scene: &mut ResolvedScene| {
            let _ = scene.get_or_insert_template::<C::Template>(context);
            scene.push_bundle_template(DedupedFieldEffect::<C, T, Src, P, S>::new(
                sources, producer, setter,
            ));
        },
    )
}

#[cfg(test)]
mod tests {
    use crate::{self as bevy_scene, bsn, reactive_field, ScenePlugin, WorldSceneExt};
    use bevy_app::{App, TaskPoolPlugin};
    use bevy_asset::AssetPlugin;
    use bevy_ecs::{prelude::*, signal::Effects};

    #[derive(Component, Default, Clone, Debug)]
    struct Health {
        current: u32,
        max: u32,
    }

    #[derive(Component, Default, Clone)]
    struct Untouched(u32);

    #[derive(Component, Default, Clone)]
    struct Label(u32);

    fn test_app() -> App {
        let mut app = App::new();
        app.add_plugins((
            TaskPoolPlugin::default(),
            AssetPlugin::default(),
            ScenePlugin,
        ));
        app
    }

    /// The hand-written form, which `$` expands to.
    #[test]
    fn reactive_field_updates_only_that_field() {
        let mut app = test_app();
        let world = app.world_mut();
        let hp = world.spawn_signal(30u32);

        let entity = world
            .spawn_scene(bsn! {
                Health { max: 100 }
                Untouched(7)
                {reactive_field(
                    hp,
                    |hp: u32| hp,
                    |health: &mut Health, value| health.current = value,
                )}
            })
            .unwrap()
            .id();

        let health = world.get::<Health>(entity).unwrap();
        assert_eq!(health.current, 30, "effect writes the initial value");
        assert_eq!(
            health.max, 100,
            "static patch on the same component survives"
        );
        assert_eq!(world.get::<Untouched>(entity).unwrap().0, 7);

        *world.signal_mut(hp).unwrap() = 55;
        world.run_effects();

        let health = world.get::<Health>(entity).unwrap();
        assert_eq!(health.current, 55);
        assert_eq!(health.max, 100, "sibling field is not clobbered");
        assert_eq!(world.get::<Untouched>(entity).unwrap().0, 7);
    }

    /// `$signal` — the bare sugar.
    #[test]
    fn reactive_signal_sugar() {
        let mut app = test_app();
        let world = app.world_mut();
        let hp = world.spawn_signal(30u32);

        let entity = world
            .spawn_scene(bsn! {
                Health { current: $hp, max: 100 }
                Untouched(7)
            })
            .unwrap()
            .id();

        assert_eq!(world.get::<Health>(entity).unwrap().current, 30);
        assert_eq!(world.get::<Health>(entity).unwrap().max, 100);

        *world.signal_mut(hp).unwrap() = 55;
        world.run_effects();
        assert_eq!(world.get::<Health>(entity).unwrap().current, 55);
        assert_eq!(world.get::<Health>(entity).unwrap().max, 100);
        assert_eq!(world.get::<Untouched>(entity).unwrap().0, 7);
    }

    /// A braced expression mentioning `$` is promoted to reactive, and the body is plain Rust
    /// over plain values.
    #[test]
    fn reactive_expression_over_multiple_signals() {
        let mut app = test_app();
        let world = app.world_mut();
        let hp = world.spawn_signal(2u32);
        let bonus = world.spawn_signal(100u32);

        let entity = world
            .spawn_scene(bsn! {
                Health { current: {$hp * 2 + $bonus}, max: 100 }
            })
            .unwrap()
            .id();

        assert_eq!(world.get::<Health>(entity).unwrap().current, 104);

        *world.signal_mut(hp).unwrap() = 5;
        world.run_effects();
        assert_eq!(world.get::<Health>(entity).unwrap().current, 110);

        *world.signal_mut(bonus).unwrap() = 200;
        world.run_effects();
        assert_eq!(world.get::<Health>(entity).unwrap().current, 210);
    }

    /// `$(self, C)` — a component on the entity being spawned, with no entity plumbing at all.
    /// This is the "react to a component I already have" case.
    #[test]
    fn reactive_own_component() {
        let mut app = test_app();
        let world = app.world_mut();

        let entity = world
            .spawn_scene(bsn! {
                Health { current: 3, max: 10 }
                Label({$(self, Health).current * 2})
            })
            .unwrap()
            .id();

        assert_eq!(world.get::<Label>(entity).unwrap().0, 6);

        // Ordinary mutation, from anywhere.
        world.get_mut::<Health>(entity).unwrap().current = 8;
        world.run_effects();
        assert_eq!(world.get::<Label>(entity).unwrap().0, 16);
    }

    /// `$(expr, C)` — a component on an entity value in scope.
    #[test]
    fn reactive_component_on_another_entity() {
        let mut app = test_app();
        let world = app.world_mut();
        let model = world
            .spawn(Health {
                current: 4,
                max: 50,
            })
            .id();

        let view = world
            .spawn_scene(bsn! {
                Health { current: {$(model, Health).current}, max: 100 }
            })
            .unwrap()
            .id();

        assert_eq!(world.get::<Health>(view).unwrap().current, 4);

        world.get_mut::<Health>(model).unwrap().current = 17;
        world.run_effects();
        assert_eq!(world.get::<Health>(view).unwrap().current, 17);
        assert_eq!(world.get::<Health>(view).unwrap().max, 100);
    }

    /// `$(#Name, C)` — a scene reference, resolved when the scene is applied. This is the case
    /// that needs the build-time `resolve` phase.
    #[test]
    fn reactive_component_on_a_scene_ref() {
        let mut app = test_app();
        let world = app.world_mut();

        let root = world
            .spawn_scene(bsn! {
                #Root
                Health { current: 7, max: 10 }
                Children [
                    ( Label({$(#Root, Health).current * 3}) )
                ]
            })
            .unwrap()
            .id();

        let child = world.get::<Children>(root).unwrap()[0];
        assert_eq!(world.get::<Label>(child).unwrap().0, 21);

        // Mutating the referenced entity drives the child's field.
        world.get_mut::<Health>(root).unwrap().current = 9;
        world.run_effects();
        assert_eq!(world.get::<Label>(child).unwrap().0, 27);
    }

    /// Mixed source kinds in one expression.
    #[test]
    fn reactive_mixed_sources() {
        let mut app = test_app();
        let world = app.world_mut();
        let bonus = world.spawn_signal(100u32);

        let entity = world
            .spawn_scene(bsn! {
                Health { current: 1, max: 10 }
                Label({$(self, Health).current + $bonus})
            })
            .unwrap()
            .id();

        assert_eq!(world.get::<Label>(entity).unwrap().0, 101);

        world.get_mut::<Health>(entity).unwrap().current = 5;
        world.run_effects();
        assert_eq!(world.get::<Label>(entity).unwrap().0, 105);

        *world.signal_mut(bonus).unwrap() = 200;
        world.run_effects();
        assert_eq!(world.get::<Label>(entity).unwrap().0, 205);
    }

    /// `$(Resource)` — a resource source. This is the case that previously tempted a system param
    /// (`Res<Theme>`), which would have read the resource *untracked* and left the field stale.
    #[test]
    fn reactive_resource_source() {
        #[derive(Resource, Default, Clone)]
        struct Scale(u32);

        let mut app = test_app();
        let world = app.world_mut();
        world.insert_resource(Scale(10));
        let hp = world.spawn_signal(3u32);

        let entity = world
            .spawn_scene(bsn! {
                Label({$hp * $(Scale).0})
            })
            .unwrap()
            .id();

        assert_eq!(world.get::<Label>(entity).unwrap().0, 30);

        // Changing the resource updates the field, with no `$` source on the signal side moving.
        world.resource_mut::<Scale>().0 = 100;
        world.run_effects();
        assert_eq!(world.get::<Label>(entity).unwrap().0, 300);

        *world.signal_mut(hp).unwrap() = 5;
        world.run_effects();
        assert_eq!(world.get::<Label>(entity).unwrap().0, 500);
    }

    /// A resource inserted *after* the effect first ran still wakes it, because the dependency is
    /// recorded even while the resource is absent.
    #[test]
    fn reactive_resource_source_inserted_late() {
        #[derive(Resource, Default, Clone)]
        struct Bonus(u32);

        let mut app = test_app();
        let world = app.world_mut();

        let entity = world
            .spawn_scene(bsn! {
                Label({$(Bonus).0 + 1})
            })
            .unwrap()
            .id();

        assert_eq!(
            world.get::<Label>(entity).unwrap().0,
            1,
            "absent reads as Default"
        );

        world.insert_resource(Bonus(41));
        world.run_effects();
        assert_eq!(world.get::<Label>(entity).unwrap().0, 42);
    }

    /// Two reactive fields on one component, each with its own source.
    #[test]
    fn two_reactive_fields_on_one_component() {
        let mut app = test_app();
        let world = app.world_mut();
        let current = world.spawn_signal(1u32);
        let max = world.spawn_signal(10u32);

        let entity = world
            .spawn_scene(bsn! {
                Health { current: $current, max: $max }
            })
            .unwrap()
            .id();

        assert_eq!(world.get::<Health>(entity).unwrap().current, 1);
        assert_eq!(world.get::<Health>(entity).unwrap().max, 10);

        *world.signal_mut(current).unwrap() = 4;
        world.run_effects();
        assert_eq!(world.get::<Health>(entity).unwrap().current, 4);
        assert_eq!(world.get::<Health>(entity).unwrap().max, 10);

        *world.signal_mut(max).unwrap() = 99;
        world.run_effects();
        assert_eq!(world.get::<Health>(entity).unwrap().current, 4);
        assert_eq!(world.get::<Health>(entity).unwrap().max, 99);
    }

    #[test]
    fn reactive_works_on_children_and_tuple_structs() {
        let mut app = test_app();
        let world = app.world_mut();
        let count = world.spawn_signal(2u32);

        let root = world
            .spawn_scene(bsn! {
                Children [ ( Untouched($count) ) ]
            })
            .unwrap()
            .id();

        let child = world.get::<Children>(root).unwrap()[0];
        assert_eq!(world.get::<Untouched>(child).unwrap().0, 2);

        *world.signal_mut(count).unwrap() = 8;
        world.run_effects();
        assert_eq!(world.get::<Untouched>(child).unwrap().0, 8);
    }

    #[test]
    fn each_spawn_gets_its_own_effect() {
        let mut app = test_app();
        let world = app.world_mut();
        let hp = world.spawn_signal(3u32);

        let scene = || bsn! { Health { current: $hp, max: 100 } };

        let a = world.spawn_scene(scene()).unwrap().id();
        let b = world.spawn_scene(scene()).unwrap().id();

        *world.signal_mut(hp).unwrap() = 12;
        world.run_effects();

        // A signal captured outside the scene is shared, but each instance has its own effect.
        assert_eq!(world.get::<Health>(a).unwrap().current, 12);
        assert_eq!(world.get::<Health>(b).unwrap().current, 12);
        assert_eq!(world.get::<Effects>(a).unwrap().len(), 1);
        assert_eq!(world.get::<Effects>(b).unwrap().len(), 1);
    }

    #[test]
    fn despawning_the_scene_entity_tears_down_its_effect() {
        let mut app = test_app();
        let world = app.world_mut();
        let hp = world.spawn_signal(1u32);

        let entity = world
            .spawn_scene(bsn! { Health { current: $hp, max: 10 } })
            .unwrap()
            .id();

        world.entity_mut(entity).despawn();

        *world.signal_mut(hp).unwrap() = 2;
        world.run_effects();
        // Nothing panics, and the effect is gone rather than writing to a dead entity.
    }
}
