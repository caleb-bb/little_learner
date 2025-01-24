#lang racket
(require malt)

;; Generally speaking, when definining a large neural network, we want to define
;; the layer functions and shapes together for each layer, and then stack the
;; layers into a single function/shapelist.
;;
;; Shape lists follow this pattern for DENSE layers:
;;
;;      width(layer_1) length(input_vector))                  LIST OF WEIGHT TENSORS
;;      length(input_vector))                                 BIAS VECTOR
;;      width(layer_3) width(layer_1))                        LIST OF WEIGHT TENSORS
;;      width(layer_3))                                       BIAS VECTOR
;;      width(layer_4) width(layer_3))                        LIST OF WEIGHT TENSORS
;;      layer_3)                                              BIAS VECTOR
;;      ...
;;      width(layer_(n-1)) width(layer_n))                    LIST OF WEIGHT TENSORS
;;      width(layer_n))                                       BIAS VECTOR
;;      length(output_vector) width(layer_n))                 LIST OF WEIGHT TENSORS
;;      length(output_vector))                                BIAS VECTOR
;;
;;
;;
;; If we have a network with many layers, and we grab a random slice of 4 layers beginning with a list of weight tensors, we get:
;;
;; (width(layer_(n-1)) width(layer_n))
;; (width(layer_(n)))
;; (width(layer_(n)) width(layer_(n+1)))
;;
;; Stripped to just numbers, we have:
;;
;;  n-1     n
;;  n
;;  n       n+1
;;  n+1
;;  n+1     n+2
;;  n+2
;;  n+2     n+3
;;
;; The point is to squeeze or expand the output into the right size for the next layer and/or output vector.
;;
;; A block is a function that takes two arguments:
;;  1. A layer function
;;  2. A shape list
;;
;; And returns a neural network layer. Bad-ass.

(define block (lambda (fn shape-lst) (list fn shape-lst)))
(define block-fn (lambda (ba) (ref ba 0)))
(define block-ls (lambda (ba) (ref ba 1)))

;; (define layer1 (block relu (list (list 64 32 (list 64)))))
;; (define layer2 (block relu (list (list 45 64 (list 45)))))
;; (define layer3 (block relu (list (list 26 45 (list 26)))))

(define block-compose
  (lambda (f g j) (lambda (t) (lambda (theta) ((g ((f t) theta)) (refr theta j))))))

;;block-comopose expects two block functions f and g as well as an integer j. j
;; is the number of parameters from theta that f will consume.

(define 3-layer-network (stack-blocks (list layer1 layer2 layer3)))

;; stack-blocks returns a new block function and a new block lst
;; the block function is just the composition of all the block functions in each layer
;; the bloc klist is just a concatenation of all the block lists
;; So in this case, the new block function would be relu(relu(relu())) and the
;; new list is (list 64 32 46 64 26 45)

(define stack2
  (lambda (ba bb)
    (block (block-compose (block-fn ba) (block-fn bb) (block-ls ba))
           (append (block-ls ba) (block-ls bb)))))

(define stack-blocks (lambda (bls) (stack-blocks (refr bls 1) (ref bls 0))))
(define stacked-blocks
  (lambda (rbls ba)
    (cond
      [(null? rbls) ba]
      [else (stacked-blocks (refr rbls 1) (stack2 ba (ref rbls 0)))])))

;; THE LAW OF BLOCKS
;; Blocks can be stacked to form bigger blocks and complete networks

(define dense-block (lambda (n m) (block relu (list (list m n) (list m)))))

;; This function defines a dense block because, in a dense block, every neuron
;; in a layer connects to every neuron in the next layer. Therefore, each layer's
;; output is a tensor with length equal to the width of the next layer, so that
;; every output of every neuron in layer n can be fed to every neuron in layer n+1.

(define layer1 (dense-block 32 64))
(define layer2 (dense-block 64 45))
(define layer3 (dense-block 45 26))
