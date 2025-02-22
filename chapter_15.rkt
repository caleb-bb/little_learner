#lang racket
(require malt)
(require malt/examples/morse)

(define corr (lambda (t) (lambda (theta) (+ (correlate (ref theta 0) t) (ref theta 1)))))

;; In corr, theta_0 is a filter bank, since we expect t to be a signal-2. This
;; makes sense, because correlation always happens between a filter bank and a
;; signal, and we see here that correlate is called with the first element of theta
;; and then t as arguments, and this is then added to theta_1. We see htat theta_1,
;; then, must be a bias parameter, meaning a tensor-1 of length equal to the filter
;; bank's length. Since the bias parameter is a tensor-1, it contains scalars.

;; This function, recu, is also known as conv, conv1D, in other neural network
;; systems. 'recu' stands for rectifying correlational unit.

(define recu (lambda (t) (lambda (theta) (rectify (corr t) theta))))

;; for recu, theta_0 must be a filter bank, which is a tensor-3, whereas theta_0
;; for relu was a tensor-2. The shape list of a recu consists of one shape for the
;; filter bank and one shape for the bias tensor. The filte rbank has the shape (b
;; m d) and the bias tensor has the shape (list b). So the shape list would be
;; (list (list b m d) (list b)). If the shape of the input signal-2 t is (list n d)
;; where (theta_0) has the shape (b m d) and theta_1 has the shape (b).
;;
;; rectify does not hange the shape of its argument. The shape of the output is
;; the same as the shape of ((corr t) theta), which is the shape of (+ (correlate
;; theta_0 t) theta_1). The shape fo the result of + there is driven by the shapeo
;; f(correlate theta_0 t) which, from the law of correlation on page 317, is (n b).

(define recu-block (lambda (b m d) (block recu (list (list b m d) (list b)))))

;; Networks that use corelation are convlolutional neural networks.
;;
;; So, wtf is recu-block doing? Well, it's simple.
;; Recall how block works. block takes a layer function (a function that returns
;; a layer of artificial neurons) and a shape list. In this case, block takes recu
;; (which is a layer function) and shape list consisting of (list (list b m d)
;; (list b))
;;
;; Much as a model is an algorithm (or target function) and a set of weights, a
;; layer is a layer function and a set of weights. This reveals an interesting and
;; deep truth about the nature of neural networks: each neuron layer is, in itself,
;; a complete model, because a layer qualifies as a neural network and has its own
;; set of weights. A neural network of more than one layer, then, can always be
;; decomposed into a set of neural networks feeding into one another. Moreover,
;; each neuron is a layer, because an artificial neuron has a set of weights and an
;; algorithm to decide things.
;;
;; Recall that correlation is the process of sliding a filter along a source
;; signal to derive a tensor of matches for that filter agaist various segments
;; against that signal. So, if we look at how corr is defined, we see that theta_0
;; is a pattern, t is a source signal, and theta_1 is a bias vector.
;;
;; So, recu here is our artificial neuron; it composes a linear function (corr)
;; with a decider function (rectify). Note that recu is also a layer function, in
;; this case, because every neuron is also a layer. When we call ((stack (block))
;; N) on a single neuron, N, that neuron goes from being a layer composed of a
;; single neuron to being a layer composed of multiple neurons.
;;
;; Accordingly, recu-block's nature becomes very obvious: it simple takes a
;; shape, and creates a block defined by recu, which, in turn, is a correlative
;; function.
;;
;; Convolutional neural networks usee convolution rather than correlation.
;; Convolution is the same thing as correlation except that the filters are
;; reversed before invoking the correlation.
;;
;; (define sum-2
;;  (lambda (t)
;;      (summed-2 t (sub1 (tlen t) 0.0))))
;;
;; (define summed-2
;;  (lambda (t i a)
;;      (cond
;;          ((zero? i) (+ (ref t 0) a))
;;          (else
;;              (summed-2 t (sub1 i) (+ (ref t i) a))))))

(define sum-2 sum-1)
(define sum-cols (ext1 sum-2 2))

;; If we take a sum and then average it by dividing the number of segments into
;; it, then we are doing what is known as global average pooling.

(define signal-avg
  (lambda (t) (lambda (theta) (/ (sum-cols t) (ref (refr (shape t) (- (rank t) 2)) 0)))))

(define signal-avg-block (block signal-avg (list)))

;; This is a fully convolutional block, meaning there are no dense layers and
;; everything except for signal-avg is is recu.
(define fcn-block (lambda (b m d) (stack-blocks (list (recu-block b m d) (recu-block b m b)))))

(define morse-fcn
  (stack-blocks (list (fcn-block 4 3 2) (fcn-block 8 3 4) (fcn-block 16 3 8) (fcn-block 26 3 16))))

(define init-shape
  (lambda (s)
    (cond
      [((= (len s 1) (zero-tensor s)))
       ((= (len s) 2) (random-tensor 0.0 (/ 2 (tref s 1) s)))
       ((= (len s) 3) (random-tensor 0.0 (/ 2 (* (tref s 1) (tref s 2)) s)))])))

(define trained-morse
  (lambda (classifier theta-shapes)
    (model classifier
           (adam-gradient-descent (sampling-obj (l2-loss classifier) morse-train-xs morse-train-ys)
                                  (init-theta theta-shapes)))))

(define train-morse
  (lambda (network)
    (with-hypers ((alpha 0.0005) (revs 2000) (batch-size 8) (mu 0.9) (beta 0.999))
                 (trained-morse (block-fn network) (block-ls network)))))
(define fcn-model (train-morse morse-fcn))

(define skip
  (lambda (f j)
    (lambda (t)
      (lambda (theta)
        (+ (f t) theta)
        ((correlate (tref theta j) t))))))
