use std::collections::BinaryHeap;

fn main() {
    mincost_to_hire_workers(vec![10, 20, 5], vec![70, 50, 30], 2);
}

pub fn mincost_to_hire_workers(quality: Vec<i32>, wage: Vec<i32>, k: i32) -> f64 {
    let n = quality.len();

    // in our selected group, we pay each worker r*quality[i], for some base rate r.
    // For every worker, to satisfy the wage constraint, we need
    // r*quality[i] >= wage[i].

    // The smallest r can be is the largest wage[i]/ quality[i] ratio.
    // Thus, it makes sense to scan workers ordered by wage[i]/quality[i] ascending.
    // as we progress, for any subset of previous workers that uses the current one (i),
    // the smallest r will be wage[i]/quality[i].

    // The total cost is then r * sum(quality of selected).
    // To minimise the total, we want to minimumse the quality of selected.
    // We do this using a max heap of qualities of size k and track the current
    // quality total.

    let mut order = (0..n).collect::<Vec<_>>();
    order.sort_by(|&x, &y| {
        let r1 = (wage[x] as f64) / (quality[x] as f64);
        let r2 = (wage[y] as f64) / (quality[y] as f64);
        r1.total_cmp(&r2)
    });

    let mut heap = BinaryHeap::<i32>::new();
    let mut q_sum = 0;
    let mut ans = f64::INFINITY;

    for i in order {
        let r = (wage[i] as f64) / (quality[i] as f64);
        let q = quality[i];
        heap.push(q);
        q_sum += q;

        if heap.len() > k as usize {
            let big_q = heap.pop().unwrap();
            q_sum -= big_q;
        }
        if heap.len() == k as usize {
            ans = ans.min(r * q_sum as f64);
        }
    }

    ans
}
