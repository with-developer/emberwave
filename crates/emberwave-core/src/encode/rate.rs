//! Rate control: how many bit-planes each 32x32 block drops so that the
//! frame fits its budget.
//!
//! A block can be sent at a few sizes. For each, it drops the number of
//! planes that gives the least distortion there: dropping a plane sometimes
//! costs no word and still lands the reconstruction nearer. From one size to
//! the next smaller one is a step, which saves words and adds distortion; its
//! key orders it by the distortion it adds per word it saves. The steps of
//! all blocks are applied in key order, ties by block index, until the frame
//! fits: the cheapest distortion goes first, frame-wide, in one pass
//! (decision D28). A block's keys never decrease, so its steps come in the
//! order they build on each other.
//!
//! Keys are integers found by exact comparisons, with no division and no
//! logarithm, so that a GPU, given the same costs, makes the same choices.

use super::quant::DROP_STEPS;

/// Number of keys.
pub(crate) const KEYS: usize = 512;

/// 2^`exponent` as an exact `f32`, for exponents in the normal range.
fn exact_pow2(exponent: i32) -> f32 {
    f32::from_bits(((127 + exponent) as u32) << 23)
}

/// The threshold of key `key`: a step reaches it when it adds at least
/// `saving * threshold` of distortion. Thresholds run from 2^-64 to 2^64 in
/// four steps per power of two, with mantissas 1, 1.25, 1.5 and 1.75: a
/// saving of up to 4095 words times a threshold is exact in `f32`.
fn threshold(key: usize) -> f32 {
    let exponent = (key / 4) as i32 - 64;
    let mantissa = (4 + key % 4) as f32 / 4.0;
    mantissa * exact_pow2(exponent)
}

/// The key of a step that adds `distortion` and saves `saving` words: the
/// highest threshold it reaches, 0 for none (including a step that adds no
/// distortion at all).
pub(crate) fn key(distortion: f32, saving: u16) -> u16 {
    let saving = f32::from(saving);
    let reaches = |key: usize| distortion >= saving * threshold(key);
    // The thresholds increase: find the first one above 0 not reached.
    let (mut low, mut high) = (1usize, KEYS);
    while low < high {
        let middle = (low + high) / 2;
        if reaches(middle) {
            low = middle + 1;
        } else {
            high = middle;
        }
    }
    (low - 1) as u16
}

/// What the rate control knows of one 32x32 block.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct BlockCost {
    /// Size in words with 0 to 15 planes dropped; 0 when nothing is left to
    /// send. Never increases with more planes dropped.
    pub words: [u16; DROP_STEPS],
    /// Weighted squared error of the coded 8x8 blocks with 0 to 15 planes
    /// dropped.
    pub distortion: [f32; DROP_STEPS],
    /// The most planes the block may drop.
    pub max_drop: u8,
}

/// A step of a block: drop planes up to `drop`, saving `saving` words.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Step {
    pub key: u16,
    pub block: u32,
    pub drop: u8,
    pub saving: u16,
}

/// The numbers of planes a block may drop that are worth considering: for
/// each size the block can have, the one with the least distortion (the
/// fewest planes on a tie). Sizes decrease along the list.
pub(crate) fn points(cost: &BlockCost) -> Vec<u8> {
    let mut points: Vec<u8> = Vec::new();
    for drop in 0..=cost.max_drop.min(DROP_STEPS as u8 - 1) {
        let (words, distortion) = (
            cost.words[usize::from(drop)],
            cost.distortion[usize::from(drop)],
        );
        match points.last_mut() {
            Some(last) if cost.words[usize::from(*last)] == words => {
                if distortion < cost.distortion[usize::from(*last)] {
                    *last = drop;
                }
            }
            _ => points.push(drop),
        }
    }
    points
}

/// Appends the steps of block `block` in the order they must be applied:
/// from each of its [`points`] to the next. A step's key is at least that of
/// the step before it.
pub(crate) fn steps(block: u32, cost: &BlockCost, out: &mut Vec<Step>) {
    let mut previous_key = 0u16;
    for pair in points(cost).windows(2) {
        let (from, to) = (usize::from(pair[0]), usize::from(pair[1]));
        let saving = cost.words[from] - cost.words[to];
        let added = cost.distortion[to] - cost.distortion[from];
        let key = key(added, saving).max(previous_key);
        out.push(Step {
            key,
            block,
            drop: pair[1],
            saving,
        });
        previous_key = key;
    }
}

/// How many planes each block drops so that all of them together take at
/// most `budget_words`, or `None` if they do not fit even with every block
/// dropping all it may.
pub(crate) fn choose(costs: &[BlockCost], budget_words: u64) -> Option<Vec<u8>> {
    // Each block starts at the best drop that costs what dropping none does.
    let mut drops: Vec<u8> = costs.iter().map(|cost| points(cost)[0]).collect();
    let total: u64 = costs.iter().map(|cost| u64::from(cost.words[0])).sum();
    let Some(need) = total.checked_sub(budget_words).filter(|&need| need > 0) else {
        return Some(drops);
    };

    let mut all = Vec::new();
    for (block, cost) in costs.iter().enumerate() {
        steps(block as u32, cost, &mut all);
    }
    all.sort_unstable_by_key(|step| (step.key, step.block, step.drop));

    let mut saved = 0u64;
    for step in all {
        if saved >= need {
            break;
        }
        drops[step.block as usize] = step.drop;
        saved += u64::from(step.saving);
    }
    (saved >= need).then_some(drops)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keys_order_distortion_per_word() {
        assert_eq!(key(0.0, 10), 0);
        assert_eq!(key(-1.0, 10), 0);
        assert_eq!(key(f32::NAN, 10), 0);
        // One word for a distortion of 1: threshold 2^0 is key 256.
        assert_eq!(key(1.0, 1), 256);
        assert_eq!(key(1.25, 1), 257);
        assert_eq!(key(1.2499, 1), 256);
        assert_eq!(key(2.0, 1), 260);
        // The same ratio gives the same key, whatever the saving.
        assert_eq!(key(300.0, 300), 256);
        assert_eq!(key(4095.0 * 1.75, 4095), 259);
        // Out of range on either side.
        assert_eq!(key(1e-30, 1), 0);
        assert_eq!(key(1e30, 1), KEYS as u16 - 1);
        // More distortion per word never gives a smaller key.
        let mut previous = 0;
        for i in 0..2000 {
            let k = key(1.01f32.powi(i) * 1e-3, 7);
            assert!(k >= previous);
            previous = k;
        }
    }

    fn cost(words: [u16; 4], distortion: [f32; 4]) -> BlockCost {
        let mut c = BlockCost {
            words: [words[3]; DROP_STEPS],
            distortion: [distortion[3]; DROP_STEPS],
            max_drop: 3,
        };
        c.words[..4].copy_from_slice(&words);
        c.distortion[..4].copy_from_slice(&distortion);
        c
    }

    #[test]
    fn a_frame_that_fits_drops_nothing() {
        let costs = [cost([10, 6, 3, 0], [0.0, 1.0, 5.0, 20.0])];
        assert_eq!(choose(&costs, 10), Some(vec![0]));
        assert_eq!(choose(&costs, 100), Some(vec![0]));
    }

    #[test]
    fn the_cheapest_distortion_per_word_goes_first() {
        // Block 0 saves 4 words for 1 (0.25 per word); block 1 saves 4 for 40.
        let costs = [
            cost([10, 6, 3, 0], [0.0, 1.0, 50.0, 500.0]),
            cost([10, 6, 3, 0], [0.0, 40.0, 50.0, 500.0]),
        ];
        // 4 words over budget: block 0's first step.
        assert_eq!(choose(&costs, 16), Some(vec![1, 0]));
        // 8 words over: both first steps, in key order.
        assert_eq!(choose(&costs, 12), Some(vec![1, 1]));
        // Everything, and not one word less.
        assert_eq!(choose(&costs, 0), Some(vec![3, 3]));
    }

    #[test]
    fn a_cheap_step_waits_for_the_dear_one_before_it() {
        // Dropping one plane is dear, the second is nearly free: the second
        // cannot go before the first, so it gets the first's key.
        let c = cost([10, 8, 2, 0], [0.0, 100.0, 100.5, 900.0]);
        let mut out = Vec::new();
        steps(0, &c, &mut out);
        assert_eq!(out.len(), 3);
        assert_eq!(out[1].key, out[0].key);
        assert!(out[2].key >= out[1].key);
        assert_eq!(out.iter().map(|s| s.drop).collect::<Vec<_>>(), [1, 2, 3]);
    }

    #[test]
    fn a_plane_that_saves_nothing_rides_with_the_next() {
        let c = cost([10, 10, 4, 0], [0.0, 3.0, 9.0, 50.0]);
        let mut out = Vec::new();
        steps(5, &c, &mut out);
        assert_eq!(out.len(), 2);
        assert_eq!((out[0].drop, out[0].saving), (2, 6));
        // Its key counts the distortion of both planes: 9 for 6 words.
        assert_eq!(out[0].key, key(9.0, 6));
    }

    #[test]
    fn at_the_same_size_the_least_distortion_wins() {
        // Dropping one plane costs no word but lands nearer, as can happen
        // with a dead zone: the block uses it even when everything fits.
        let c = cost([4, 4, 4, 0], [2.0, 0.1, 13.8, 40.0]);
        assert_eq!(points(&c), [1, 3]);
        assert_eq!(choose(std::slice::from_ref(&c), 100), Some(vec![1]));
        let mut out = Vec::new();
        steps(0, &c, &mut out);
        assert_eq!(out.len(), 1);
        assert_eq!((out[0].drop, out[0].saving), (3, 4));
        assert_eq!(out[0].key, key(39.9, 4));
    }

    #[test]
    fn an_impossible_budget_is_none() {
        let mut c = cost([10, 6, 3, 1], [0.0, 1.0, 5.0, 20.0]);
        c.max_drop = 3;
        assert_eq!(choose(&[c], 0), None);
    }
}
