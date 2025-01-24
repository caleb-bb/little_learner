#lang racket
(require malt)
(require malt/examples/iris)

(define next-a
  (lambda (t i a)
    (cond
      [(> (tref t i) (tref t a)) i]
      [else a])))

(define argmaxed
  (lambda (t i a)
    (let ([alpha-hat (next-a t i a)])
      (cond
        [(zero? i) alpha-hat]
        [else (argmaxed t (sub1 i) alpha-hat)]))))

(define argmax-1 (lambda (t) (let ([i (sub1 (tlen t))]) (argmaxed t i i))))

(define class=1
  (lambda (t u)
    (cond
      [(= (argmax-1 t) (argmax-1 u)) 1.0]
      [else 0.0])))

(define class= (ext2 class=1 1 1))

(define accuracy (lambda (a-model xs ys) (/ (sum (class= (a-model xs) ys)) (tlen xs))))
(define dense-block (lambda (n m) (block relu (list (list m n) (list m)))))
(define iris-network (stack-blocks (list (dense-block 4 6) (dense-block 6 3))))
(define iris-classifier (block-fn iris-network))
(define iris-theta-shapes (block-ls iris-network))

(define accurate-enough-iris-theta?
  (lambda (theta) (>= (accuracy (model iris-classifier theta) iris-test-xs iris-test-ys))))

(grid-search accurate-enough-iris-theta?
             ((revs 500 1000 2000 4000) (alpha 0.0001 0.0002 0.0005) (batch-size 4 8 16))
             (naked-gradient-descent
              (sampling-obj (l2-loss iris-classifier) iris-train-xs iris-train-ys)
              (init-theta iris-theta-shapes)))

;; grid-search just lets us try different hyperparameters
