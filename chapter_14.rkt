#lang racket
(require malt)

;; We can define a filter as a short signal used for pattern matching. All
;; filters are shorter than the source and their length is generally an odd number.
;;
;; For a filter of length n, we take a slice of the source beginning at index i
;; and ending at i+(n-1). We take the dot product of those two, of the slice and
;; pattern like so.
;;
;; SOURCE:      [0.0 0.0 0.0 1.0 0.0 ...]
;; PATTERN:     [0.0 1.0 0.0]
;;
;; SLICE:       [0.0 0.0 0.0]
;; PATTERN:     [0.0 1.0 0.0]
;;
;; DOT PRODUCT:
;;      1. (dot-product [0.0 0.0 0.0] [0.0 1.0 0.0])
;;      2. (sum (* [0.0 0.0 0.0] [0.0 1.0 0.0]))
;;      3. (sum [0.0 0.0 0.0])
;;      4. 0.0
;;
;; The resulting scalar tells us how similar the overlap is.
;;
;; Once more, beginning at index 2:
;;
;; SOURCE:      [0.0 0.0 0.0 1.0 0.0]
;; PATTERN:             [0.0 1.0 0.0]
;;
;; SLICE:       [0.0 1.0 0.0]
;; PATTERN:     [0.0 1.0 0.0]
;;
;; DOT PRODUCT:
;;      1. (dot-product [0.0 1.0 0.0] [0.0 1.0 0.0])
;;      2. (sum (* [0.0 1.0 0.0] [0.0 1.0 0.0]))
;;      3. (sum [0.0 1.0 0.0])
;;      4. 1.0
;;
;;  A positive value! There's overlap here.
;;
;; Now, we can slide the pattern all the way along the source, collecting the
;; resulting scalars into a vector having length 1 + (n - m) where n is the length
;; of the source and m is the length of the pattern. This process is known as
;; correlation, and the vector thus produced is known as a result signal.
;; Rigorously, the correlation between a given a source and a pattern is a vector
;; of all dot products between that pattern and n-legth slices of that source,
;; where n is the length of the pattern. A "slice" here is understood as a vector
;; of consecutive entries.
;;
;; We can match border cases by padding a vector with 0.0 on either end. We can
;; match border cases for higher-ranking signals by padding using higher-ranking
;; tensors. The padding will always be a tensor having rank one less than the rank
;; of the signal. Padding is important because, later on, we want to be able to
;; produce result tensors having the same depth as the original signal. By padding
;; the signal and then zipping the result tensors for all patterns, we can produce
;; a tensor with the appropriate shape.
;;
;; We can also CASCADE correlations. To cascade correlations means to correlate
;; a pattern to a source, then take the resulting vector and use it as source for
;; another pattern, and then keep doing that as much as we can.
;;
;; THE LAW OF CORRELATION (single filter)
;; The correlation of a filter of length m with a souce signal-1 of length n,
;; where m is odd (given by 2p + 1), is a signal of length n obtained by sliding
;; the filter from overlap position -p to overlap position n-p-1, where each
;; segment of the result signal-1 is obtained by taking the dot product of the
;; filter and the overlap in the source at each overlap position.
;;
;; Generally speaking, the first correlation (or set thereof) will find
;; primitive features, while successive correlations will find more abstract and
;; high-level features. This parallels how features are matched as they pass
;; through the layers of a neural net.
;;
;; We would like for our neural network to derive its own filters from the data.
;; Since theta is mutated with each revision, it stands to reason that anything
;; derived must have a place in theta. Ergo, filters are tensor parameters in a
;; theta.
;;
;; Correlation may be localized to a neuron layer. A layer performing this task
;; is called a correlation layer. We learn filters by trining the network on a
;; data set where each x is a signal-1. The parameters reflecting the filters are
;; filter weights. We learn the filters by using correlation functions to identify
;; patterns in the xs that yield the appropriate ys.
;;
;; If we have one source and b filters, and each filter has length m, then there
;; exists a tensor-2 known as a filter bank having shape (b m) and consisting of
;; all those filters.
;;
;; If we correlate b tensors with a source of length n, then we have b tensors,
;; each of which has length n. Remember that zipping two tensors together is like
;; transposing a matrix; it basically rotates the whole thing 90 degree
;; counter-clockwise. And we want to be ready for cascading. So we take our b
;; tensors of length n and zip them pairwise, producing n tensors of length b. This
;; effectively changes the shape from (b n) to (n b), allowing us to cascade to
;; another correlation layer of equal width.
;;
;; Let's upscale to a signal-2. Let there be a signal-2 with shape (n d). Any
;; filter used on that signal-2 must have depth d. This is because the dot product
;; always operates along some axis, and requires the two tensors it has as
;; arguments to be the same length along that axis. This is why we pad.
;;
;; To recap, we correlate our signal-2 using these steps:
;;      0. Define a signal-2 of shape (n d)
;;      1. Define our filters so that their depth is equal to the depth of the signal-2.
;;      2. Our filter bank has shape (b m d ) where b is the number of filters, m
;;          is the number of segments (width) of each filter, and d is the depth of each
;;          filter (which is also the depth of the source signal-2).
;;      3. Pad the source signal-2 using signal-1s on either end so that we can
;;          correlate and get n results, where n is the length of the signal-2. This gives
;;          us a result tensor of shape (b n d)
;;      4. Transpose that tensor by zipping all b tensors together, along the
;;          axis having length n, yielding a tensor of shape (n b).
;;          (notice how zipping decreases the rank of a tensor)
;;      5. Feed that result into the next correlation layer! :-)
;;
;; THE LAW OF CORRELATION (filter bank)
;; The correlation of a filter bank of shape (b m d) with a source signal-2 of
;; shape (n d) is a signal-2 of shape (n b) resulting from zipping the b signal-1s
;; resulting from correlating the b filters-2 in the bank with the source.
