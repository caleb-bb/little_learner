#lang racket
(require malt)

(+ (tensor 2 3) (tensor 5 7))
(+ (tensor (tensor (tensor 2 3) (tensor 5 7))) (tensor (tensor 11 13) (tensor 17 19)))

;; The sum of any natural number of tensors is a single tensor with the same shape, where corresponding scalars are added.
;; The + operator descends into the tensors. We can visualize it "splitting", like so:
;; (+ (tensor 2 3) (tensor 5 7))
;; (tensor (+ 2 5) (+ 3 7))
;; Let there be n tensors of rank r and shape [s1, s2, s3 ... sr].
;; (+ t1, t2, t3 .... tn) is a single tensor having the same shape and rank, with these values:
;; ((+ t1.1, t2.1, ... t[sZ).1), (+ t1.2, t2.2, t2.2 ... t[sZ].2) ... (+ t[sZ-k].n, t[sZ-k+1].n, t[sZ-k+2].3, ... t[sZ].n))
;; Where t1, t2, t3 etc are the tensors, up to tn
;; Where Z is the current depth in the shape list, i.e. the first layer would have a final term at s1, the second at s2, etc.a
;; The sum of a tensor and a scalar is the sum of all scalars in that tensor with that scalar.
;; This is the POINTWISE EXTENSION of addition
;;
;; There are similar pointwise extensions of other operators, such as square roots.

(define sum1
  (lambda (t)
    (summed t (sub1 (tlen t) 0.0))))

(define summed
  (lambda (t i a)
    (cond
      ((zero? i) (+ (tref t 0) a))
      (else (summed t (sub1 i) (+ (tref t i) a   ))))))

;; The pointwise extension of the sum function is sum*. We say that sum* takes a single tensor of rank r and returns a tensor of rank (r-1).
;;
;; 1. (tensor (line (tensor 2 7 5 11) (list 4 6)))
;; 2. (tensor (line-2 4 6) (line-7 4 6) (line-5 4 6) (line-11 4 6))
;; 3. (tensor 14 34 26 50)
