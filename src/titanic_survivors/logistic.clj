(ns titanic-survivors.logistic
  (:require
    [incanter.core :as i]
    [incanter.optimize :as o]
    [incanter.stats :as s]))


;; Logistic regression

(defn logistic-class
  "Logistic classifier, one or zero!"
  [x]
  (if (>= x 0.5) 1 0))

(defn sigmoid-function
  "Returns a sigmoidal function of x for use in logistic regression.
  Positive beta gives a greater probability of positive class,
  negative beta gives a greater probability of negative class
  (given a positive x)."
  [beta]
  (let [beta-trans (i/trans beta)
        z  (fn [x] (- (first (i/mmult beta-trans x))))]
    (fn [x]
      (/ 1
         (+ 1
            (i/exp (z x)))))))

(defn logistic-cost
  "Calculates the overall (average) cost for a for a given vector of
  coefficients y-hats and a vector of training data ys."
  [ys y-hats]
  (let [cost (fn [y y-hat]
               (if (zero? y)
                 (- (i/log (- 1 y-hat)))
                 (- (i/log y-hat))))]
    (s/mean (map cost ys y-hats))))

(defn logistic-regression
  "Returns the logistic-cost value based on the provided coefficients.
  The initial coefficients are 0.0 for each parameter."
  [ys xs]
  (let [cost-fn (fn [coefs]
                  (let [classify (sigmoid-function coefs)
                        y-hats   (map (comp classify i/trans) xs)]
                    (logistic-cost ys y-hats)))
        init-coefs (repeat (i/ncol xs) 0.0)]
    (o/minimize cost-fn init-coefs)))