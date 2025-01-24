#lang racket
(require malt)

;; THE LAW OF ZIPPED SIGNALS
;; A signal-2 is formed by zipping signals-1, and the signal-2 as well as its constituent signals-1 all have the same number of segments.

(tensor (tensor 2 3 5) (tensor 7 11 13))

;; A signal is a variation in a physical quantity over time. A continuous signal
;; is analog. If we want to decode signals with a neural net, then we have to find
;; a way to turn them into tensors, much as we have to do with anything else that
;; we use a neural net for.
;;
;; One way to encode a signal as a tensor is to break the signal into discrete
;; chunks of time and assign a scalar to each chunk. For example, we could break
;; the signal into second-by-second chunks and record a scalar for the strength of
;; the physical quantity at that time. After a number of seconds, we will have a
;; tensor. Here, it would be a vector.
;;
;; A signal-1 is a one-dimensional signal which is also a one-dimensional
;; tensor. A signal-2 is a two-dimensional signal which is also a two-dimensional
;; tensor. We get a signal-2 by zipping two signal-1s together.
;;
;; We can think of it almost like transposing a matrix. Two signal-1s:
;; [2  3  5  7  11  13]
;; [17 19 23 29 31  37]
;;
;; Zipped:
;; [[2  17]
;;  [3  19]
;;  [5  23]
;;  [7  29]
;;  [11 31]
;;  [13 37]]
