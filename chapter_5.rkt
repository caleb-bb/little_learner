#lang racket
(require malt)

(declare-hyper revs)
(declare-hyper alpha)

;; (define old-gradient-descent
;;   (lambda (obj theta)
;;     (let ((f (lambda (big_theta)
;;                (map (lambda (p g)
;;                       (- p (* 0.1 g)))
;;                     big_theta
;;                     (gradient-of obj big_theta)))))
;;                     (revise f 1000 theta))))

;; (define old-l2-loss
;;   (lambda (target)
;;     (lambda (xs ys)
;;       (lambda (theta)
;;         (let ((pred-ys ((target xs) theta)))
;;           (sum (sqr (- ys pred-ys))))))))

(define crap-gradient-descent
  (lambda (obj theta revs-arg)
    (with-hypers ((revs revs-arg))
    (let ((f (lambda (big_theta alpha-arg)
               (with-hypers ((alpha alpha-arg))
               (map (lambda (p g)
                      (- p (* alpha g)))
                    big_theta
                    (gradient-of obj big_theta))))))
      (revise f revs theta)))))

(define quad-xs (tensor -1.0 0.0 1.0 2.0 3.0))
(define quad-ys (tensor 2.55 2.1 4.35 10.2 18.25))

;; To scale is to multiply by a scalar.
;; A linear function uses only addition and scaling. I suspect that such
;; functions can be represented as polynomials having degree less than 2.

(define quad
  (lambda (t)
    (lambda (theta)
      (+ (* (ref theta 0) (sqr t))
         (+ (* (ref theta 1) t) (ref theta 2))))))


(define l2-loss
  (lambda (target)
    (lambda (xs ys)
    (lambda (theta)
      (let ((pred-ys ((target xs) theta)))
        (sum (sqr (- ys pred-ys))))))))

(define sum-1
  (lambda (t)
    (summed t (sub1 (tensor t) 0.0))))

(define summed
  (lambda (t i a)
    (cond
      ((zero? i) (+ (tref t 0) a))
      (else
       (summed t (sub1 i) (+ (trefs t i) (+ (trefs t) a)))))))

;; (define obj ((l2-loss quad) quad-xs quad-ys))

(define revise
  (lambda (f revs theta)
    (cond
      ((zero? revs) theta)
      (else
       (revise f (sub1 revs) (f theta))))))

(define gradient-descent
  (lambda (obj theta)
    (let ((f (lambda (big_theta)
               (map (lambda (p g)
                      (- p (* alpha g)))
                         big_theta
                         (gradient-of obj big_theta)))))
          (revise f revs theta))))

;; (gradient-descent)
;; => (lambda obj theta)
;; => (lambda big_theta)
;; (gradient-descent ARG)

;; (with-hypers ((revs 1000) (alpha 0.001))
;;   (gradient-descent
;;    ((l2-loss quad) quad-xs quad-ys)
;;                    (list 0.0 0.0 0.0)))

(define plane-xs
  (tensor
   (tensor 1.0 2.05)
   (tensor 1.0 3.0)
   (tensor 2.0 2.0)
   (tensor 2.0 3.9)
   (tensor 3.0 6.13)
   (tensor 4.0 8.09)))

(define plane-ys (tensor 13.99 15.99 18.0 22.4 30.2 37.94))

(define plane
  (lambda (t)
    (lambda (theta)
      (+
       (dot-product (ref theta 0) t)
       (ref theta 1)))))

(define dot-product-rough
  (lambda (w t)
    (sum-1 (* w t))))

;; THE RULE OF DATA SETS
;; In a data set (xs, ys), both xs and ys must have the same number of elements.
;; The elments of xs, howevever, can have a different shape from the elements of
;; ys.
;;
;; THE RULE OF PARAMETERS
;; Every parameter is a tensor.
;;
;; THE RULE OF THETA
;; Theta is a list of parameters that can have different shapes

(with-hypers
  ((revs 1000) (alpha 0.001))
  (gradient-descent ((l2-loss plane) plane-xs plane-ys)
                    (list (tensor 0.0 0.0) 0.0)))


;; (define gradient-descent
;;   (lambda (obj theta)
;;     (let ((f (lambda (big_theta)
;;                (map (lambda (p g)
;;                       (- p (* alpha g)))
;;                          big_theta
;;                          (gradient-of obj big_theta)))))
;;           (revise f revs theta))))

;; (gradient-of ((l2-loss plane) plane-xs plane-ys) (list (tensor 0.0 0.0) 0.0))

;; (l2-loss plane)
;;  => (lambda (target) plane (...))
;;  => (lambda (xs ys) (...))
;;  => (lambda (theta) (...))
;;  => (let ((pred-ys ((plane xs) theta))) (sum (sqr (- ys pred-ys))))
;;  == (let ((pred-ys ((plane xs) theta))) (sum (sqr (- ys pred-ys))))

;; (plane xs)
;; => (lambda (t) (...) xs)
;; => (+ (dot-product (tref theta 0) xs) (tref theta 1))

;; ((plane xs) theta) WHERE  theta = (list (tensor 0.0 0.0) 0.0)
;; => (+ (dot-product (tref (list (tensor 0.0 0.0) 0.0) 0) xs) (tref theta 1))

;; (dot-product (tref theta 0) xs)
;; => (lambda (w t) (tref theta 0) xs)
;; == (lambda (w t) ())
