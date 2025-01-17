#lang racket
(require malt)

(define smooth (lambda (decay-rate average g) (+ (* decay-rate average) (* (- 1.0 decay-rate) g))))

;; Every reference to a "scalar" below is a special case. Our smooth function is
;; defined using extended operators, so it can operate on tensors of any rank.

;; average here is a "historically acucmulated average" or "historical average".
;; The decay-rate is a scalar between 0 and 1 that will probably turn out to be a
;; hyperparameter. The point of smoothing is to take a bunch of scalars and move
;; them all close to one another. If we visualize them as y-values on a graph where
;; the x-axis is their positional value within the vector, then we can visual the
;; dots lying closer to a straight line. Note that in the degenerate case here,
;; where decay rate == 0, the smoothing function is equivalent to just subtrating
;; the gradient from the average, which is always the previous value in the vector
;; post-smoothing.

;; The purpose of the decay rate is to reduce how much a given scalar
;; contributes to the historical average with each subsequent application of the
;; smoothing function. Each time we apply the function, only 90% of the historical
;; average is used. So the further apart two scalars are in a sequence, the less
;; the earlier scalar influences the later one. Thus, the influence of the early
;; scalars DECAYS as time goes by.

;; This rersults in a true SMOOTHING function.

;; Suppose we begin with a historical average of
;;  (tensor 0.8 3.1 2.2)
;;  And then we have these tensors:
;;  (tensor 1.0 1.1 3.0)
;;  (tensor 13.4 18.2 41.2)
;;  (tensor 1.1 0.3 67.3)
;;
;; If we run them all through our smoothing function, using the result each time
;; as the new average, we get:
;;
;; (tensor 0.82 2.9 2.28)
;; (tensor 2.078 4.43 6.172)
;; (tensor 1.98 4.01 12.28)

;; The overall plan of the LL seems to be:
;; 1. Introduce the necessary math
;; 2. Teach the reader gradient descent using step-by-step function composition
;; 3. Begin adding various optimizations to gradient descent
