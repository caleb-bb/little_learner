#lang racket
(require malt/base)
;; There are a multitude of ways to change the velocity of revision. One of them is
;; momentum. Another way, however, is to multiply the learning rate by some factor
;; D. Remember that the learning rate effectively takes a chunk out of the gradient
;; and uses that as a velocity. We don't want to change alpha directly because
;; alpha (the learning rate) is a hyperparameter and should not be mutated during
;; use.

;; Keep in mind that the gradient shrinks as we approach the local minimum. Since
;; the gradient shrinks, we can make sure that the fraction of it that we use
;; shrinks more slowly than the gradient itself, to keep the speed from dropping
;; off too quickly.

;; A good way to do this is make D vary inversely as the gradient. So as the
;; gradient shrinks, D increases, multiplies by the learning rate, and then
;; multiplies by the gradient itself.

;; Let there be some factor G, such that D = 1/G and G depends on the gradient
;; and its history. This is another way of doing momentum, in some sense, because
;; we're incorporating the history of the gradient.

;; It follows that (= (* alpha D) (* alpha (/ 1 G)) (/ alpha G))
(declare-hyper beta)
(define eta 1e-08)
(define plane-xs
  (tensor (tensor 1.0 2.05)
          (tensor 1.0 3.0)
          (tensor 2.0 2.0)
          (tensor 2.0 3.9)
          (tensor 3.0 6.13)
          (tensor 4.0 8.09)))

(define plane-ys (tensor 13.99 15.99 18.0 22.4 30.2 37.94))

;; Before, we used velocity, which took a historical average of how much was
;; subtracted from the weight the last time. For rms (root mean square), we do the
;; same thing, but we use a smoothing function. All that rigmarole about D was
;; just a way to implement smoothing.
;;
;; So, from the top: in the context of gradient descent, VELOCIT is understood to
;; mean the amount we change a weight during any given revision. For example, if we
;; revise some weight, w_n, by 0.0251, then the velocity of revision is 0.0251 at
;; that revision. Simple.  Gradient descent has this problem where the velocity
;; drops as we approach the minimum representing best fit. That's because weights
;; are revised by an amount equal to the gradient multiplied by the learning rate,
;; and the gradient, in turn, depends on the derivative of the loss function, whose
;; output shrinks as we get closer to a minimum. So, we need some way to fix that.
;; Momentum to the rescue! If we take some fraction (* mu v), where v is the
;; velocity of the previous revision, and add it to the velocity of our current
;; revision, we can keep it from slowing down so fast.  The RMSprop verrsion of
;; gradient descent is a more sophisticated version of the same thing. Instead of
;; using just the previous velocity, we use an "accumulated historical average" of
;; the velocity. We do this by defining a function that takes each new velocity and
;; factors it into the amoutn of "momentum" being applied to each revision. We also
;; have a "decay rate", which is a scalar. The decay rate is part of a clever
;; function that slowly reduces the influence of a given revision's velocity with
;; each new revision. So the first revision's velocity will maatter a great deal
;; for the mometum of the second revision, but then it will matter less for the
;; second, and even less for the third, and so on.
;;
;; What does that clever function look like? Well, that's our smoothing function.
;;
;; Let's define our smoothing function: (+ (* decay-rate average) (* (- 1.0 decay-rate) g))
;; At any given revision, we take the decay-rate (which is a constant), and
;; multiply it by whatever the average velocity is at that revision. The decay
;; rate is often about .9. Call this step the "attenuated average" (my term). The
;; point of multiplying the historical average by a decay rate is to make sure that
;; data points further back in the sequence don't have too much of an influence,
;; since this is supposed to be about SMOOTHING, not forcing everything to fall
;; directly on a straight line.
;;
;; Then, we take the decay rate (which is always between 0 and 1) and subtract
;; it from 1, then mulitply that difference by the current gradient. We subtract
;; the decay rate from 1 because we don't want to add too big a chunk of the
;; gradient, as this may cause us to overshoot.
;;
;; RMSprop is similar, but with a few key differences.
;;
;; First of all, it hands the square of the gradient to the smoothing function,
;; rather than just the gradient. That square-smoothed velocity is called r.
;;
;; Second, RMSprop effectively mutates the learning rate. I say "effectively"
;; because, while the learning rate remains unchanged, RMSprop divides the learning
;; rate by the square root of r. Notice the process here: the historical average at
;; each step uses the square of the gradient to get the smoothed velocity, and then
;; we take the square root of that historical average before applying the learning
;; rate to the smoothed velocity. The result of dividing the smoothed velocity into
;; the learning rate (also called alpha) is "alpha-hat", to denote that this
;; quantity is derived from alpha. Finally, we multiply alpha-hat by the gradient
;; to get the actual velocity that is passed forward. This enables us to
;; incorporate the SMOOTHED history of velocities at each revision step.
(define rms-u
  (lambda (big-p g)
    (let ([r (smooth beta (ref big-p 1) (sqr g))])
      (let ([alpha-hat (/ alpha (+ (sqrt r) eta))]) (list (- (ref big-p 0) (* alpha-hat g)) r)))))

(define rms-i (lambda (p) (list p (zeroes p))))
(define rms-d (lambda (big-p) (ref big-p 0)))
(define rms-gradient-descent (gradient-descent rms-i rms-d rms-u))

(define try-plane
  (lambda (a-gradient-descent a-revs an-alpha)
    (with-hypers ((revs a-revs) (alpha an-alpha) (batch-size 4))
                 (a-gradient-descent (sampling-obj (l2-loss plane) plane-xs plane-ys)
                                     (list (tensor 0.0 0.0) 0.0)))))

;; (with-hypers ((beta 0.9)) (try-plane rms-gradient-descent 3000 0.1))

;; And, finally, adam-u. adam-u combines TWO smoothed historical averages to change
;; the velocity at each revision step. The gradient is smoothed using the mu
;; huperparameter, which means we have momentum. The hyperparameter, beta, is used
;; to incorporate root mean square of the historically accumulated average
;; gradient..

(define adam-u
  (lambda (big-p g)
    (let ([r (smooth beta (ref big-p 2) (sqr g))])
      (let ([alpha-hat (/ alpha (+ (sqrt r) eta))]
            [v (smooth mu (ref big-p 1) g)])
        (list (- (ref big-p 0) (* alpha-hat v)) v r)))))

(define adam-i (lambda (p) (let ([v (zeroes p)]) (let ([r v]) (list p v r)))))
(define adam-d (lambda (big-p) (ref big-p 0)))
(define adam-gradient-descent (gradient-descent adam-i adam-d adam-u))

;; Compare this to

(define old-velocity-u
  (lambda (big_p g) (let ([v (- (* mu (ref big_p 1) (* alpha g)))]) [list (+ (ref big_p 0) v) v])))

;; Notice two things remain the same.
;; First, r is still the smoothed square of the historical average gradient, as in rms-u.
;; Second, alpha-hat is still a mutated derivation of the learning rate, as in rms-u
;;
;; Notice that the accumulation is different here. In old-velocity-u, we set v
;; equal to the difference between (* big_p_one mu) and (* alpha g). In adam-u, we
;; instead pass mu, big_p_1, and (* alpha g) to smooth. So mu is our new decay
;; rate, big_p_1 acts as an average (which is multiplied with mu) and the gradient
;; passed in is NOT multiplied by the learning rate. Instead, the learning arte
;; is used to derive alpha-hat, which is finally multiplied by v before becoming
;; the accompaniment for the parameter next iteration.
