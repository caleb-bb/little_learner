#lang racket
(require malt)

(define line
  (lambda (x)
    (lambda (theta)
      (+ (* (ref theta 0) x) (ref theta 1)))))

(define sum
  (lambda (t)
    (summed t (sub1 (tlen t)) 0.0)))

(define summed
  (lambda (t i a)
    (cond
      ((zero? i) (+ (tref t 0) a))
      (else (summed t (sub1 i) (+ (tref t i) a   ))))))

(define loss
  (lambda (actual)
    (lambda (arguments parameters)
      (sum (sqr (- actual ((line arguments) parameters)))))))

(define l2-loss
  (lambda (target)
    (lambda (xs ys)
      (lambda (theta)
        (let ((pred-ys ((target xs) theta)))
          (sum (sqr (- ys pred-ys))))))))

(define line-xs (tensor 2.0 1.0 4.0 3.0))
(define line-ys (tensor 1.8 1.2 4.2 3.3))
;; (define theta (list 0.0 0.0))

;; ((l2-loss line) line-xs line-ys)
;;  ((l2-loss line) (tensor 2.0 1.0 4.0 3.0) (tensor 1.8 1.2 4.2 3.3))
;;  (lambda (theta) (let (pred-ys (line (tensor 2.0 1.0 4.0 3.0) theta))))
;;  (sum (sqr (- (tensor 1.8 1.2 4.2 3.3) pred-ys)))
;;  Notice that this is an objective function, because it expects a single value and returns a scalar.
(define obj ((l2-loss line) line-xs line-ys))

;; Recall how rate-of-change works. Rate-of-change w.r.t theta_0 is given by (L2
;; - L1)/(new_theta_0 - old_theta_0), or dL/dtheta. It's just the change in
;; whatever the loss fuction returns divided by the change in theta
;;
;; We're trying to find a local minima for the loss function. So if we graph the
;; loss function - say it's a two dimensional and a parabola - then the rate of
;; change at theta_0 (where theta_0 is understood to be represented by the x-axis)
;; is tangent to that parabola.
;;
;; Since the rate of change always gives the slope of a tangent line, then where
;; the rate of change = 0, we've got ourselves an at least local minimum.
;;
;; The slope of the tangent line to the loss curve at a given theta_n is the gradient for that theta.

;; (gradient-of (lambda (theta) (sqr (ref theta 0))) (list 27.0))
;; because the gradient tells us the rate of change in the loss curve, we can
;; iterate with it. We can take that rate of change and use it to adjust our
;; theta_n based on the gradient for that theta_n at whatever value it has.

(define revise
  (lambda (f revs theta)
    (cond
      ((zero? revs) theta)
      (else
       (revise f (sub1 revs) (f theta))))))


;; where f is whatever function we use to revise theta
;; So suppose that f is
;;  (lambda (theta)
;;      (map (lambda (p)
;;          (- p 3)
;;          theta))
;;
;; and also let revs = 5 and theta = (list 1 2 3)
;;
;; The we have
;;
;; (revise f 5 (list 1 2 3))
;; (revise f 4 (list -2 -1 0))
;; (revise f 3 (list -5 -4 -3))
;; (revise f 2 (list -8 -7 -6))
;; (revise f 1 (list -11 -10 -9))
;; (revise f 0 (list -14 -13 -12))
;; (list -14 -13 -12)


;; LAW OF REVISION - FINAL VERSION
;; new theta_i = theta_i - (learning_rate * gradient w.r.t. theta_i)
;; This is what's happening to each p in theta on line 3
;; 1. (let ((f (lambda (theta)
;; 2.           (let ((gs (gradient-of (obj theta)))
;; 3.                 (map (lambda (p g) (- p (* alpha g))) theta gs)))))
;; 4.      (revise f 1000 (list 0.0 0.0))))

(define alpha 0.01)
(define revs 1000)

(define gradient-descent
  (lambda (obj theta)
    (let ((f (lambda (big_theta)
               (map (lambda (p g)
                      (- p (* alpha g)))
                    big_theta
                    (gradient-of obj big_theta)))))
          (revise f revs theta))))
