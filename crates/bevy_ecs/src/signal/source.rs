//! Reactive sources: the things a reactive field can depend on.
//!
//! This module is primarily the surface the `bsn!` macro's `$` syntax targets. Its items are
//! deliberately *not* re-exported from [`signal`](super) or the prelude — whether any of this
//! should become a general-purpose free-standing API is a separate decision.
//!
//! ## Two phases
//!
//! A source is read on every effect re-run, but some sources need data that is only available once,
//! while the scene is being applied. A `#Name` scene reference is the motivating case: it is an
//! [`EntityTemplate`] that only becomes an [`Entity`] inside
//! [`build_template`](crate::template::Template::build_template), via
//! [`TemplateContext::get_entity`].
//!
//! So [`ReactiveSource`] is split:
//! - [`ReactiveSource::resolve`] runs **once**, during scene application, with a
//!   [`TemplateContext`]. Its output is stored in the effect.
//! - [`ReactiveSource::read`] runs on **every** effect run, with an [`EffectContext`], and records
//!   the dependency.
//!
//! The two contexts never coexist — `TemplateContext` holds an `&mut EntityWorldMut` and is gone
//! long before the effect re-runs — which is why this cannot be a single method.
//!
//! Sources that need no build-time data (a [`Signal`], or a component on the effect's own entity)
//! simply use `Resolved = ()`.

use super::{EffectContext, Signal};
use crate::{
    component::Component,
    entity::Entity,
    error::Result,
    resource::Resource,
    template::{EntityTemplate, Template, TemplateContext},
};
use core::marker::PhantomData;
use variadics_please::all_tuples_enumerated;

/// Something a reactive field can depend on.
///
/// See the [module docs](self) for why this has two phases.
pub trait ReactiveSource: Clone + Send + Sync + 'static {
    /// The value handed to the reactive expression.
    type Output: Send + Sync + 'static;

    /// Data resolved once, while the scene is applied. Use `()` if none is needed.
    type Resolved: Send + Sync + 'static;

    /// Resolves build-time data for this source. Called once per spawned entity.
    fn resolve(&self, context: &mut TemplateContext) -> Result<Self::Resolved>;

    /// Reads the current value and records a dependency on it. Called on every effect run.
    fn read(&self, resolved: &Self::Resolved, context: &mut EffectContext) -> Self::Output;
}

/// A [`Signal`] is a source with no build-time data.
///
/// Yields the signal's value, or [`Default`] if the signal entity has been despawned.
impl<T: Clone + Default + Send + Sync + 'static> ReactiveSource for Signal<T> {
    type Output = T;
    type Resolved = ();

    fn resolve(&self, _context: &mut TemplateContext) -> Result<Self::Resolved> {
        Ok(())
    }

    fn read(&self, _resolved: &Self::Resolved, context: &mut EffectContext) -> Self::Output {
        context.track(*self).unwrap_or_default()
    }
}

/// Component `C` on the entity the effect is attached to.
///
/// This needs no entity plumbing at all: the effect already knows its own target.
pub struct Own<C>(PhantomData<fn() -> C>);

impl<C> Default for Own<C> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

impl<C> Clone for Own<C> {
    fn clone(&self) -> Self {
        Self(PhantomData)
    }
}

impl<C: Component + Clone + Default> ReactiveSource for Own<C> {
    type Output = C;
    type Resolved = ();

    fn resolve(&self, _context: &mut TemplateContext) -> Result<Self::Resolved> {
        Ok(())
    }

    fn read(&self, _resolved: &Self::Resolved, context: &mut EffectContext) -> Self::Output {
        let Some(target) = context.target() else {
            return C::default();
        };
        context.track_component::<C>(target).unwrap_or_default()
    }
}

/// Component `C` on a specific [`Entity`] that is already known when the scene is built.
///
/// The entity slot is an [`EntityTemplate`], so this covers both a plain `Entity` value and a
/// `#Name` scene reference — the latter is resolved during scene application, exactly as `#Name`
/// values in ordinary component fields are.
pub struct On<C>(pub EntityTemplate, PhantomData<fn() -> C>);

impl<C> On<C> {
    /// Tracks `C` on the entity described by `entity`.
    pub fn new(entity: impl Into<EntityTemplate>) -> Self {
        Self(entity.into(), PhantomData)
    }
}

impl<C> Clone for On<C> {
    fn clone(&self) -> Self {
        Self(self.0, PhantomData)
    }
}

impl<C: Component + Clone + Default> ReactiveSource for On<C> {
    type Output = C;
    type Resolved = Entity;

    fn resolve(&self, context: &mut TemplateContext) -> Result<Self::Resolved> {
        self.0.build_template(context)
    }

    fn read(&self, resolved: &Self::Resolved, context: &mut EffectContext) -> Self::Output {
        // A forward `#Name` reference can resolve to an entity whose components have not been
        // written yet. `track_component` records the dependency even when the component is absent,
        // so the effect re-runs once it is inserted.
        context.track_component::<C>(*resolved).unwrap_or_default()
    }
}

/// Resource `R` as a reactive source.
///
/// Resources carry change ticks like components do, so this slots into the same machinery. Making
/// resources first-class sources is what keeps things like a theme or settings resource *tracked*
/// — reading one through a system param instead would leave the field silently stale when it
/// changes.
pub struct OnResource<R>(PhantomData<fn() -> R>);

impl<R> Default for OnResource<R> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

impl<R> Clone for OnResource<R> {
    fn clone(&self) -> Self {
        Self(PhantomData)
    }
}

impl<R: Resource + Clone + Default> ReactiveSource for OnResource<R> {
    type Output = R;
    type Resolved = ();

    fn resolve(&self, _context: &mut TemplateContext) -> Result<Self::Resolved> {
        Ok(())
    }

    fn read(&self, _resolved: &Self::Resolved, context: &mut EffectContext) -> Self::Output {
        // The dependency is recorded even when the resource is absent, so inserting it later
        // wakes the effect.
        context.track_resource::<R>().unwrap_or_default()
    }
}

/// One or more [`ReactiveSource`]s: what a reactive field actually depends on.
///
/// Implemented for tuples and, like [`Bundle`](crate::bundle::Bundle), for a single source on its
/// own — so `hp` and `(hp,)` are both valid, the former yielding `T` where the latter yields
/// `(T,)`.
///
/// Resolving and reading happen for the whole collection at once, so the reactive expression can be
/// a *pure* function of the read values — it never touches an [`EffectContext`] itself.
pub trait ReactiveSources: Clone + Send + Sync + 'static {
    /// The values handed to the reactive expression.
    type Output: Send + Sync + 'static;

    /// The build-time resolved data.
    type Resolved: Send + Sync + 'static;

    /// Resolves every source. Called once per spawned entity.
    fn resolve_all(&self, context: &mut TemplateContext) -> Result<Self::Resolved>;

    /// Reads every source, recording a dependency on each.
    fn read_all(&self, resolved: &Self::Resolved, context: &mut EffectContext) -> Self::Output;
}

/// A single source is a collection of one, yielding its value directly rather than a 1-tuple.
///
/// This mirrors [`Bundle`](crate::bundle::Bundle), where a lone component is a valid bundle.
impl<S: ReactiveSource> ReactiveSources for S {
    type Output = S::Output;
    type Resolved = S::Resolved;

    fn resolve_all(&self, context: &mut TemplateContext) -> Result<Self::Resolved> {
        self.resolve(context)
    }

    fn read_all(&self, resolved: &Self::Resolved, context: &mut EffectContext) -> Self::Output {
        self.read(resolved, context)
    }
}

macro_rules! reactive_sources_impl {
    ($(($index: tt, $source: ident, $alias: ident)),*) => {
        #[expect(
            clippy::allow_attributes,
            reason = "This is a tuple-related macro; as such, the lints below may not always apply."
        )]
        #[allow(
            clippy::unused_unit,
            reason = "The zero-length tuple produces a `()` expression."
        )]
        impl<$($source: ReactiveSource),*> ReactiveSources for ($($source,)*) {
            type Output = ($($source::Output,)*);
            type Resolved = ($($source::Resolved,)*);

            fn resolve_all(&self, _context: &mut TemplateContext) -> Result<Self::Resolved> {
                Ok(($(self.$index.resolve(_context)?,)*))
            }

            fn read_all(
                &self,
                _resolved: &Self::Resolved,
                _context: &mut EffectContext,
            ) -> Self::Output {
                ($(self.$index.read(&_resolved.$index, _context),)*)
            }
        }
    };
}

all_tuples_enumerated!(reactive_sources_impl, 0, 12, S, s);
