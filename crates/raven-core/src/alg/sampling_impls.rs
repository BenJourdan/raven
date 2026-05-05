
use itertools::izip;
use anyhow::{Result, anyhow};

use super::{
    DynamicClustering,
    SamplingInfo,   
};
use crate::types::{
    Contribution, 
    FDelta, 
    FloatScalar, 
    HB, 
    HS, 
    NonStrict, 
    NonStrictCarrierOps, 
    SmoothedContribution, 
    Strict, 
    StrictCarrierOps, 
    TreeIndex, 
    Volume, 
    WrapsCarrierFloat
};



impl<const ARITY: usize, V, T> DynamicClustering<ARITY, V, T> 
where
    V: std::hash::Hash + Eq + Clone + Copy,
    T: FloatScalar, // T must be a floating point type (either f32 or f64)
    Strict<T>: StrictCarrierOps<Scalar = T> + WrapsCarrierFloat,
    NonStrict<T>: NonStrictCarrierOps<Scalar = T> + WrapsCarrierFloat,   
{

    /// Get the base f contribution for a node.
    #[inline(always)]
    pub fn f_b(&self, node_idx: TreeIndex, info: &SamplingInfo<V, T>) -> Strict<T> {
        // f_b = sigma* size + (sigma * vol)/deg(x^*)

        let size = (Strict::<T>::from_non_zero_usize(self.tree_data.size[node_idx])).into_scalar();
        let vol  = self.tree_data.volume[node_idx].0.into_scalar();
        let sigma = info.sigma.into_scalar();
        let sigma_over_x_star_deg = info.sigma_over_x_star_deg.into_scalar();
        
        unsafe{
            Strict::from_positive_scalar_unchecked(
                sigma.mul_add(size, sigma_over_x_star_deg * vol)
            )
        }
    }

    // Get the delta f contribution for a node.
    #[inline(always)]
    pub fn f_delta_read(&self, node_idx: TreeIndex, info: &SamplingInfo<V, T>) -> FDelta<T> {
        // return saved f_delta if timestamps match, else return 0.
        if self.tree_data.timestamp[node_idx] == info.timestamp {
            self.tree_data.f_delta[node_idx]
        }else{
            FDelta::zero()
        }
    }

    /// Get the f contribution for a node
    #[inline(always)]
    pub fn f(&self, node_idx: TreeIndex, info: &SamplingInfo<V,T >) -> Contribution<T> {
        // f_s = f_b - f_delta
        let f_b = self.f_b(node_idx, info).into_scalar();
        let f_delta = self.f_delta_read(node_idx, info).into_float().into_scalar();

        Contribution(
            unsafe{
                NonStrict::from_positive_scalar_unchecked((f_b - f_delta).max(T::zero()))
            }
        )

    }

    #[inline(always)]
    pub fn contribution_from_arrays(
        // A fused version of f() to compute contributions for all children of a parent node
        // and write them into an output buffer.
        &self,
        output_buffer: &mut [NonStrict<T>; ARITY],
        parent_idx: TreeIndex,
        info: &SamplingInfo<V, T>,
    ) -> usize {
        let start = self.child_index(parent_idx, 0).0;
        let end = (start + ARITY).min(self.tree_data.size.len());

        let sizes = &self.tree_data.size[start..end];
        let volumes = &self.tree_data.volume[start..end];
        let saved_f_deltas = &self.tree_data.f_delta[start..end];
        let saved_timestamps = &self.tree_data.timestamp[start..end];

        let cur_timestamp = info.timestamp;
        let sigma_over_x_star_deg = info.sigma_over_x_star_deg;
        let sigma = info.sigma;

        let filled = end - start;

        // clear output buffer in the unused portion:
        output_buffer[filled..].fill(NonStrict::from(0.0));

        for (o, s, v, f_del, t) in izip!(
            &mut output_buffer[..filled],
            sizes,
            volumes,
            saved_f_deltas,
            saved_timestamps
        ) {
            let size_f = Strict::from(*s as f64);
            let vol_f = v.0;
            let f_delta_f = f_del.0 * Strict::from((*t == cur_timestamp) as u8);

            let total = sigma.mul_add(size_f, sigma_over_x_star_deg.mul_add(vol_f, -f_delta_f));
            *o = NonStrict::max(total, NonStrict::from(0.0));
        }
        filled
    }

    #[inline(always)]
    pub fn h_b(&self, node_idx: TreeIndex, info: &SamplingInfo<V, T>) -> HB<T> {
        let saved_timestamp = self.tree_data.timestamp[node_idx];
        let cur_timestamp = info.timestamp;
        let saved_h_b = self.tree_data.h_b[node_idx];
        let vol = self.tree_data.volume[node_idx].0;

        // If timestamps match, return saved h_b. Else, return vol.
        if saved_timestamp == cur_timestamp {
            saved_h_b
        } else {
            HB(vol.into())
        }
    }

    #[inline(always)]
    pub fn h_s(&self, node_idx: TreeIndex, info: &SamplingInfo<V, T>) -> HS<T> {
        let saved_timestamp = self.tree_data.timestamp[node_idx];
        let cur_timestamp = info.timestamp;
        let saved_h_s = self.tree_data.h_s[node_idx];

        // If timestamps match, return saved h_s. Else, return 0.
        if saved_timestamp == cur_timestamp {
            saved_h_s
        } else {
            HS(0.0.into())
        }
    }

    #[inline(always)]
    pub fn g(&self, node_idx: TreeIndex, info: &SamplingInfo<V, T>) -> SmoothedContribution<T> {
        // g = f(S)/f(X) + h_b(S)/w(C(x^*)) + h_s(S)

        let f_s = self.f(node_idx, info).0;
        let total_contribution_inv = info.total_contribution_inv.unwrap().0;
        let x_star_seed_set_volume_inv = info.x_star_seed_set_volume_inv;
        let h_b = self.h_b(node_idx, info).0;
        let h_s = self.h_s(node_idx, info).0;

        SmoothedContribution(f_s.mul_add(
            total_contribution_inv,
            h_b.mul_add(x_star_seed_set_volume_inv, h_s),
        ))
    }

    #[inline(always)]
    pub fn smoothed_contribution_from_arrays(
        // A fused version of g() to compute smoothed contributions for all children of a parent node
        // and write them into an output buffer.
        &self,
        output_buffer: &mut [NonStrict<T>; ARITY],
        parent_idx: TreeIndex,
        info: &SamplingInfo<V, T>,
    ) -> usize {
        let start = self.child_index(parent_idx, 0).0;
        let end = (start + ARITY).min(self.tree_data.size.len());

        let sizes = &self.tree_data.size[start..end];
        let volumes = &self.tree_data.volume[start..end];
        let saved_f_deltas = &self.tree_data.f_delta[start..end];
        let saved_h_bs = &self.tree_data.h_b[start..end];
        let saved_h_ss = &self.tree_data.h_s[start..end];
        let saved_timestamps = &self.tree_data.timestamp[start..end];

        let cur_timestamp = info.timestamp;
        let sigma_over_x_star_deg = info.sigma_over_x_star_deg;
        let x_star_seed_set_volume_inv = info.x_star_seed_set_volume_inv;
        let total_contribution_inv = info.total_contribution_inv.unwrap().0;
        let sigma = info.sigma;

        let filled = end - start;

        // clear output buffer in the unused portion:
        output_buffer[filled..].fill(NonStrict::from(0.0));

        for (o, s, v, f_del, h_b, h_s, t) in izip!(
            &mut output_buffer[..filled],
            sizes,
            volumes,
            saved_f_deltas,
            saved_h_bs,
            saved_h_ss,
            saved_timestamps
        ) {
            let size_f = Strict::from(*s as f64);
            let vol_f = v.0;
            let f_delta_f = f_del.0 * NonStrict::from((*t == cur_timestamp) as u8);
            let h_b_f = if *t == cur_timestamp { h_b.0 } else { vol_f };
            let h_s_f = if *t == cur_timestamp {
                h_s.0
            } else {
                NonStrict::from(0.0)
            };

            let f_s = sigma.mul_add(size_f, sigma_over_x_star_deg.mul_add(vol_f, -f_delta_f));

            let total = f_s.mul_add(
                total_contribution_inv,
                h_b_f.mul_add(x_star_seed_set_volume_inv, h_s_f),
            );
            *o = NonStrict::max(total, NonStrict::from(0.0));
        }
        filled
    }

    fn sample_impl(
        &mut self,
        info: &SamplingInfo<V, T>,
        rng: &mut impl rand::Rng,
        fill: impl Fn(&Self, &mut [NonStrict<T>; ARITY], TreeIndex, &SamplingInfo<V, T>) -> usize,
    ) -> Result<(V, TreeIndex, NonStrict<T>)> {
        if self.tree_data.size.is_empty() {
            return Err(anyhow!("Cannot sample from an empty tree."));
        }

        let mut cur = TreeIndex(0);
        let mut prob = Float::from(1.0f64);
        let mut buffer = [Float::from(0.0f64); ARITY];
        let mut cdf_buffer = [Float::from(0.0f64); ARITY];

        while self.tree_data.size[cur] > 1 {
            // cur corresponds to an internal node

            // populate buffer with contributions of children
            let filled = fill(self, &mut buffer, cur, info);

            let child_contribution_sum: Float = buffer[..filled].iter().sum();
            debug_assert!(
                child_contribution_sum.is_finite(),
                "child contribution sum must be finite"
            );
            if child_contribution_sum == Float::from(0.0f64) {
                return Err(anyhow!(
                    "Cannot sample from a tree with zero total contribution."
                ));
            }
            let sample = rng.random_range(0.0..child_contribution_sum.0);
            cdf_buffer[..filled].copy_from_slice(&buffer[..filled]);

            for i in 1..filled {
                cdf_buffer[i] += cdf_buffer[i - 1];
            }

            // Now we sample a child
            let child_idx = cdf_buffer[..filled]
                .iter()
                .position(|&x| x.0 >= sample)
                .ok_or(anyhow!("Failed to sample a child node."))?;
            prob *= buffer[child_idx] / child_contribution_sum;
            cur = self.child_index(cur, child_idx);
        }

        let node_id = self.tree_to_node_map.get(&cur).unwrap();
        Ok((*node_id, cur, prob))
    }

    pub fn sample(
        &mut self,
        info: &SamplingInfo<V>,
        rng: &mut impl rand::Rng,
    ) -> Result<(V, TreeIndex, Float)> {
        self.sample_impl(info, rng, |this, buf, parent, info| {
            this.contribution_from_arrays(buf, parent, info)
        })
    }

    pub fn sample_smoothed(
        &mut self,
        info: &SamplingInfo<V>,
        rng: &mut impl rand::Rng,
    ) -> Result<(V, TreeIndex, Float)> {
        self.sample_impl(info, rng, |this, buf, parent, info| {
            this.smoothed_contribution_from_arrays(buf, parent, info)
        })
    }

    #[inline(always)]
    pub fn weighted_kernel_distance(deg_v: Float, w: EdgeWeight) -> Contribution {
        // get the contribution of u w.r.t v.
        // If v is being added, this is for computing the updated contribution of u.
        // w Delta(u,v) = w(u,v)/ deg(v)
        debug_assert!(
            deg_v != Float::from(0.0),
            "deg_v must be non-zero for weighted_kernel_distance"
        );
        (w.0 * deg_v.inv()).into()
    }
}
