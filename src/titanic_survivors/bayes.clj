(ns titanic-survivors.bayes)


(defn inc-class-total
  "Given a model and class, increments the number :n in that class."
  [model class]
  (update-in model [:classes class :n] (fnil inc 0)))

(defn inc-predictors-count-fn
  "Given a row in the model and a class, returns a function which increments
  the count of the number of times that attribute has appeared for that class."
  [row class]
  (fn [model attr]
    (let [val (get row attr)]
      (update-in model [:classes class :predictors attr val] (fnil inc 0)))))

(defn assoc-row-fn
  "Given a class attribute and predictors, returns a function which takes
  a model and a row and keeps count of the number of times each parameter
  is seen for each class label."
  [class-attr predictors]
  (fn [model row]
    (let [class (get row class-attr)]
      (reduce (inc-predictors-count-fn row class)
              (inc-class-total model class)
              predictors))))

(defn bayes-classifier
  "A naive Bayes classifier to find the frequency of classes for the
  given set of predictors and dataset."
  [class-attr predictors dataset]
  (reduce (assoc-row-fn class-attr predictors) {:n (count dataset)} dataset))

(defn posterior-probability
  "Calculates the conditional probability (posterior probability) for the provided class attribute."
  [model test class-attr]
  (let [observed (get-in model [:classes class-attr])
        prior    (/ (:n observed)
                    (:n model))]
    (apply * prior
           (for [[predictor value] test]
             (/ (get-in observed [:predictors predictor value])
                (:n observed))))))

(defn bayes-classify
  "Calculates the probability of the test input against each of the model's classes."
  [model test]
  (let [probability (partial posterior-probability model test)
        classes     (keys (:classes model))]
    (apply max-key probability classes)))