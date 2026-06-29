use anyhow::{Result, anyhow};
use itertools::izip;
use rand::RngExt;

use super::{SamplingInfo, TreeLayout, TrialWorkspace};
use crate::types::{
    Contribution, EdgeWeight, FDelta, FloatScalar, HB, HS, NodeDegree, NonStrict,
    NonStrictCarrierOps, SmoothedContribution, Strict, StrictCarrierOps, TreeIndex,
};

impl<const ARITY: usize, V, T> TrialWorkspace<'_, ARITY, V, T>
where
    V: std::hash::Hash + Eq + Clone + Copy,
    T: FloatScalar, // T must be a floating point type (either f32 or f64)
    Strict<T>: StrictCarrierOps<Scalar = T> + Copy,
    NonStrict<T>: NonStrictCarrierOps<Scalar = T> + Copy,
{
    /// Get the base f contribution for a node.
    #[inline(always)]
    pub fn f_b(&self, node_idx: TreeIndex, info: &SamplingInfo<T>) -> Strict<T> {
        // f_b = sigma* size + (sigma * vol)/deg(x^*)

        let size = (Strict::<T>::from_non_zero_usize(self.persistent.size[node_idx])).into_scalar();
        let vol = self.persistent.volume[node_idx].0.into_scalar();
        let sigma = info.sigma.into_scalar();
        let sigma_over_x_star_deg = info.sigma_over_x_star_deg.into_scalar();

        unsafe {
            Strict::from_positive_scalar_unchecked(sigma.mul_add(size, sigma_over_x_star_deg * vol))
        }
    }

    // Get the delta f contribution for a node.
    #[inline(always)]
    pub fn f_delta_read(&self, node_idx: TreeIndex, info: &SamplingInfo<T>) -> FDelta<T> {
        // return saved f_delta if timestamps match, else return 0.
        if self.query_time.timestamp[node_idx] == info.timestamp {
            self.query_time.f_delta[node_idx]
        } else {
            FDelta::zero()
        }
    }

    /// Get the f contribution for a node
    #[inline(always)]
    pub fn f(&self, node_idx: TreeIndex, info: &SamplingInfo<T>) -> Contribution<T> {
        // f_s = f_b - f_delta
        let f_b = self.f_b(node_idx, info).into_scalar();
        let f_delta = self.f_delta_read(node_idx, info).into_scalar();

        Contribution(unsafe {
            NonStrict::from_non_negative_scalar_unchecked((f_b - f_delta).max(T::zero()))
        })
    }

    #[inline(always)]
    pub fn contribution_from_arrays(
        // A fused version of f() to compute contributions for all children of a parent node
        // and write them into an output buffer.
        &self,
        output_buffer: &mut [NonStrict<T>; ARITY],
        parent_idx: TreeIndex,
        info: &SamplingInfo<T>,
    ) -> usize {
        let start = TreeLayout::<ARITY>::child_index(parent_idx, 0).0;
        let end = (start + ARITY).min(self.persistent.size.len());

        let sizes = &self.persistent.size[start..end];
        let volumes = &self.persistent.volume[start..end];
        let saved_f_deltas = &self.query_time.f_delta[start..end];
        let saved_timestamps = &self.query_time.timestamp[start..end];

        let cur_timestamp = info.timestamp;
        let sigma_over_x_star_deg = info.sigma_over_x_star_deg;
        let sigma = info.sigma;

        let filled = end - start;

        // clear output buffer in the unused portion:
        output_buffer[filled..].fill(NonStrict::zero());

        for (o, s, v, f_del, t) in izip!(
            &mut output_buffer[..filled],
            sizes,
            volumes,
            saved_f_deltas,
            saved_timestamps
        ) {
            // SAFETY:
            let size_f = Strict::from_non_zero_usize(*s).into_scalar();
            let vol_f = v.0.into_scalar();
            let f_delta_f = f_del.0.into_scalar() * T::from_bool(*t == cur_timestamp);

            let total = sigma.into_scalar().mul_add(
                size_f,
                sigma_over_x_star_deg
                    .into_scalar()
                    .mul_add(vol_f, -f_delta_f),
            );
            *o = NonStrict::from_non_negative_scalar(total.max(T::zero()))
                .expect("total must be non-negative after clamping");
        }
        filled
    }

    #[inline(always)]
    pub fn h_b(&self, node_idx: TreeIndex, info: &SamplingInfo<T>) -> HB<T> {
        let saved_timestamp = self.query_time.timestamp[node_idx];
        let cur_timestamp = info.timestamp;
        let saved_h_b = self.query_time.h_b[node_idx];
        let vol = self.persistent.volume[node_idx].0;

        // If timestamps match, return saved h_b. Else, return vol.
        if saved_timestamp == cur_timestamp {
            saved_h_b
        } else {
            HB::from_scalar(vol.into_scalar()).expect("Volume must be non-negative")
        }
    }

    #[inline(always)]
    pub fn h_s(&self, node_idx: TreeIndex, info: &SamplingInfo<T>) -> HS<T> {
        let saved_timestamp = self.query_time.timestamp[node_idx];
        let cur_timestamp = info.timestamp;
        let saved_h_s = self.query_time.h_s[node_idx];

        // If timestamps match, return saved h_s. Else, return 0.
        if saved_timestamp == cur_timestamp {
            saved_h_s
        } else {
            HS(NonStrict::zero())
        }
    }

    #[inline(always)]
    pub fn g(&self, node_idx: TreeIndex, info: &SamplingInfo<T>) -> SmoothedContribution<T> {
        // g = f(S)/f(X) + h_b(S)/w(C(x^*)) + h_s(S)

        let f_s = self.f(node_idx, info).0.into_scalar();
        let total_contribution_inv = info.total_contribution_inv.unwrap().0.into_scalar();
        let x_star_seed_set_volume_inv = info.x_star_seed_set_volume_inv.into_scalar();
        let h_b = self.h_b(node_idx, info).0.into_scalar();
        let h_s = self.h_s(node_idx, info).0.into_scalar();

        SmoothedContribution(
            NonStrict::from_non_negative_scalar(f_s.mul_add(
                total_contribution_inv,
                h_b.mul_add(x_star_seed_set_volume_inv, h_s),
            ))
            .expect("Smoothed contribution must be non-negative"),
        )
    }

    #[inline(always)]
    pub fn smoothed_contribution_from_arrays(
        // A fused version of g() to compute smoothed contributions for all children of a parent node
        // and write them into an output buffer.
        &self,
        output_buffer: &mut [NonStrict<T>; ARITY],
        parent_idx: TreeIndex,
        info: &SamplingInfo<T>,
    ) -> usize {
        let start = TreeLayout::<ARITY>::child_index(parent_idx, 0).0;
        let end = (start + ARITY).min(self.persistent.size.len());

        let sizes = &self.persistent.size[start..end];
        let volumes = &self.persistent.volume[start..end];
        let saved_f_deltas = &self.query_time.f_delta[start..end];
        let saved_h_bs = &self.query_time.h_b[start..end];
        let saved_h_ss = &self.query_time.h_s[start..end];
        let saved_timestamps = &self.query_time.timestamp[start..end];

        let cur_timestamp = info.timestamp;
        let sigma_over_x_star_deg = info.sigma_over_x_star_deg.into_scalar();
        let x_star_seed_set_volume_inv = info.x_star_seed_set_volume_inv.into_scalar();
        let total_contribution_inv = info.total_contribution_inv.unwrap().into_scalar();
        let sigma = info.sigma.into_scalar();

        let filled = end - start;

        // clear output buffer in the unused portion:
        output_buffer[filled..].fill(NonStrict::zero());

        for (o, s, v, f_del, h_b, h_s, t) in izip!(
            &mut output_buffer[..filled],
            sizes,
            volumes,
            saved_f_deltas,
            saved_h_bs,
            saved_h_ss,
            saved_timestamps
        ) {
            let size_f = Strict::from_non_zero_usize(*s).into_scalar();
            let vol_f = v.0.into_scalar();
            let f_delta_f = f_del.0.into_scalar() * T::from_bool(*t == cur_timestamp);
            let h_b_f = if *t == cur_timestamp {
                h_b.into_scalar()
            } else {
                vol_f
            };
            let h_s_f = if *t == cur_timestamp {
                h_s.into_scalar()
            } else {
                T::zero()
            };

            let f_s = sigma
                .mul_add(size_f, sigma_over_x_star_deg.mul_add(vol_f, -f_delta_f))
                .max(T::zero());

            let total = f_s.mul_add(
                total_contribution_inv,
                h_b_f.mul_add(x_star_seed_set_volume_inv, h_s_f),
            );
            *o = NonStrict::from_non_negative_scalar(total.max(T::zero()))
                .expect("smoothed contribution must be non-negative");
        }
        filled
    }

    fn sample_impl(
        &mut self,
        info: &SamplingInfo<T>,
        rng: &mut impl rand::Rng,
        fill: impl Fn(&Self, &mut [NonStrict<T>; ARITY], TreeIndex, &SamplingInfo<T>) -> usize,
    ) -> Result<(V, TreeIndex, NonStrict<T>)> {
        if self.persistent.size.is_empty() {
            return Err(anyhow!("Cannot sample from an empty tree."));
        }

        let mut cur = TreeIndex(0);
        let mut prob =
            NonStrict::from_non_negative_scalar(T::ONE).expect("one must be non-negative");
        let mut buffer = [NonStrict::zero(); ARITY];
        let mut cdf_buffer = [T::ZERO; ARITY];

        while self.persistent.size[cur].get() > 1 {
            // cur corresponds to an internal node

            // populate buffer with contributions of children
            let filled = fill(self, &mut buffer, cur, info);

            let child_contribution_sum: T = buffer[..filled].iter().map(|x| x.into_scalar()).sum();
            debug_assert!(
                child_contribution_sum.is_finite(),
                "child contribution sum must be finite"
            );
            if child_contribution_sum == T::ZERO {
                return Err(anyhow!(
                    "Cannot sample from a tree with zero total contribution."
                ));
            }
            let sample = T::from(rng.random::<f64>())
                .expect("random f64 sample should convert to scalar")
                * child_contribution_sum;
            for (cdf, contribution) in cdf_buffer[..filled].iter_mut().zip(&buffer[..filled]) {
                *cdf = contribution.into_scalar();
            }

            for i in 1..filled {
                cdf_buffer[i] = cdf_buffer[i] + cdf_buffer[i - 1];
            }

            // Now we sample a child
            let child_idx = cdf_buffer[..filled]
                .iter()
                .position(|&x| x >= sample)
                .ok_or_else(|| anyhow!("Failed to sample a child node."))?;
            let prob_scalar =
                prob.into_scalar() * buffer[child_idx].into_scalar() / child_contribution_sum;
            prob = NonStrict::from_non_negative_scalar(prob_scalar)
                .expect("sampling probability must be non-negative");
            cur = TreeLayout::<ARITY>::child_index(cur, child_idx);
        }

        let node_id = self.tree_to_node_map.get(&cur).unwrap();
        Ok((*node_id, cur, prob))
    }

    pub fn sample(
        &mut self,
        info: &SamplingInfo<T>,
        rng: &mut impl rand::Rng,
    ) -> Result<(V, TreeIndex, NonStrict<T>)> {
        self.sample_impl(info, rng, |this, buf, parent, info| {
            this.contribution_from_arrays(buf, parent, info)
        })
    }

    pub fn sample_smoothed(
        &mut self,
        info: &SamplingInfo<T>,
        rng: &mut impl rand::Rng,
    ) -> Result<(V, TreeIndex, NonStrict<T>)> {
        self.sample_impl(info, rng, |this, buf, parent, info| {
            this.smoothed_contribution_from_arrays(buf, parent, info)
        })
    }

    #[inline(always)]
    pub fn weighted_kernel_distance(deg_v: NodeDegree<T>, w: EdgeWeight<T>) -> Contribution<T> {
        // get the contribution of u w.r.t v.
        // If v is being added, this is for computing the updated contribution of u.
        // w Delta(u,v) = w(u,v)/ deg(v)
        let deg_v = deg_v.into_scalar();
        debug_assert!(
            deg_v != T::ZERO,
            "deg_v must be non-zero for weighted_kernel_distance"
        );
        Contribution::from_scalar(w.into_scalar() / deg_v)
            .expect("weighted kernel distance must be non-negative")
    }
}
