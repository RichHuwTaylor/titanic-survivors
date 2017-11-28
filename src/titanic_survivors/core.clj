(ns titanic-survivors.core
  (:require
    [incanter.core :as i]
    [titanic-survivors.bayes :as bayes]
    [titanic-survivors.data :as data]
    [titanic-survivors.decision :as decision]
    [titanic-survivors.logistic :as logistic]
    [titanic-survivors.statistics :as statistics]))


(defn predict-logistic-regression-test-values
  "Trains a logistic regression classifier, then uses it make predictions on the test set."
  []
  (let [train-data    (data/matrix-dataset :train)
        test-data     (data/matrix-dataset :test)
        passenger-ids (i/$ 0 test-data)
        ys            (i/$ 0 train-data)
        train-xs      (i/$ [:not 0] train-data)
        test-xs       (i/$ [:not 0] test-data)
        coefs         (:value (logistic/logistic-regression ys train-xs))
        classifier    (comp logistic/logistic-class
                            (logistic/sigmoid-function coefs)
                            i/trans)
        y-hats        (map classifier test-xs)
        predictions   (->> (i/matrix [passenger-ids y-hats])
                           (i/trans)
                           (i/dataset ["PassengerId" "Survived"]))]
    (data/save-predictions predictions)))

(defn run-logistic-regression
  "Runs logistic regression on the titanic dataset. 0 selects the first
  column of the of the matrix (the labels) [:not 0] selects the features.
  The returned :value key are the best estimates for the coefficients."
  []
  (let [data      (data/matrix-dataset :train)
        ys        (i/$ 0 data)
        xs        (i/$ [:not 0] data)]
    (logistic/logistic-regression ys xs)))

(defn test-logistic-regression-classifier
  []
  "Trains a classifier, then test it by comparing predictions to labels
  for the entire dataset. Returns a map of frequencies of correct predictions."
  (let [data       (data/matrix-dataset :train)
        ys         (i/$ 0 data)
        xs         (i/$ [:not 0] data)
        coefs      (:value (logistic/logistic-regression ys xs))
        classifier (comp logistic/logistic-class
                         (logistic/sigmoid-function coefs)
                         i/trans)
        y-hats (map classifier xs)]
    (frequencies (map = y-hats (map int ys)))))

(defn titanic-confusion-matrix
  []
  (let [data       (data/matrix-dataset :train)
        ys         (i/$ 0 data)
        xs         (i/$ [:not 0] data)
        coefs      (:value (logistic/logistic-regression ys xs))
        classifier (comp logistic/logistic-class
                         (logistic/sigmoid-function coefs)
                         i/trans)
        y-hats (map classifier xs)]
    (statistics/confusion-matrix (map int ys) y-hats)))

(defn titanic-kappa-statistic
  []
  (let [data       (data/matrix-dataset :train)
        ys         (i/$ 0 data)
        xs         (i/$ [:not 0] data)
        coefs      (:value (logistic/logistic-regression ys xs))
        classifier (comp logistic/logistic-class
                         (logistic/sigmoid-function coefs)
                         i/trans)
        y-hats     (map classifier xs)]
    (float (statistics/kappa-statistic (map int ys) y-hats))))

(defn bayes-model-survival-sex-class
  "Returns a Bayes model showing a map of predictors for the perished (n)
  and survived (y) classes."
  []
  (let [model (->> (data/load-data (:train data/file-paths))
                   (:rows)
                   (bayes/bayes-classifier :Survived [:Sex :Pclass]))]
    (println "Third class male:"
             (bayes/bayes-classify model {:Sex "male" :Pclass 3}))
    (println "First class female:"
             (bayes/bayes-classify model {:Sex "female" :Pclass 1}))))

(defn check-third-class-male-first-class-female
  "Check predictions of the Bayes classifier for the third-class male
  and first class female."
  []
  (->> (data/load-data (:train data/file-paths))
       (:rows)
       (bayes/bayes-classifier :Survived [:Sex :Pclass])
       (clojure.pprint/pprint)))

(defn test-naive-bayes-classifier
  "Compares the predictions of the naive Bayes classifier to actual survival
  in the training data."
  []
  (let [data    (:rows (data/load-data (:train data/file-paths)))
        model   (bayes/bayes-classifier :Survived [:Sex :Pclass] data)
        test    (fn [test]
                (= (:Survived test)
                   (bayes/bayes-classify model
                                   (select-keys test [:Sex :Pclass]))))
        results (frequencies (map test data))]
    (/ (get results true)
       (apply + (vals results)))))

(defn predict-naive-bayes-test-values
  "Uses the naive Bayes classifier to make predictions on the test set."
  []
  (let [train-data    (data/load-data (:train data/file-paths))
        test-data     (data/load-data (:test data/file-paths))
        passenger-ids (i/$ 0 test-data)
        train-rows    (:rows train-data)
        test-rows     (:rows test-data)
        model         (bayes/bayes-classifier :Survived [:Sex :Pclass] train-rows)
        predict       (fn [test]
                        (bayes/bayes-classify model
                                              (select-keys test [:Sex :Pclass])))
        results       (map predict test-rows)
        predictions   (->> (i/matrix [passenger-ids results])
                           (i/trans)
                           (i/dataset ["PassengerId" "Survived"])) ]
    (data/save-predictions predictions)))

(defn sex-class-decision-tree
  "Returns a decision tree for the :Pclass, :Sex and :Age predictors.
  :Age is split into three categories."
  []
  (let [data (data/load-data (:train data/file-paths))]
    (->> (i/transform-col data :Age data/age-categories)
         (:rows)
         (decision/decision-tree :Survived [:Pclass :Sex :Age])
         (clojure.pprint/pprint))))

(defn second-class-male-child-predictor
  "Returns the survival classification for a third class male
  child passenger."
  []
  (let [data (data/load-data (:train data/file-paths))
        tree (->> (i/transform-col data :Age data/age-categories)
                  (:rows)
                  (decision/decision-tree :Survived [:Pclass :Sex :Age]))
        test {:Sex "male" :Pclass 2 :Age "child"}]
    (decision/tree-classify tree test)))

(defn predict-decision-tree-test-values
  "Uses the naive Bayes classifier to make predictions on the test set."
  []
  (let [train-data    (-> (data/load-data (:train data/file-paths))
                          (i/transform-col :Age data/age-categories)
                          (i/transform-col :Fare data/fare-categories))
        test-data     (-> (data/load-data (:test data/file-paths))
                          (i/transform-col :Age data/age-categories)
                          (i/transform-col :Fare data/fare-categories))
        passenger-ids (i/$ 0 test-data)
        train-rows    (:rows train-data)
        test-rows     (:rows test-data)
        tree          (decision/decision-tree :Survived [:Pclass :Sex :Age :Fare] train-rows)
        results (->>  (map (partial decision/tree-classify tree) test-rows)
                      (map #(if (nil? %) 0 %)))
        predictions   (->> (i/matrix [passenger-ids results])
                           (i/trans)
                           (i/dataset ["PassengerId" "Survived"]))]
    (data/save-predictions predictions)))

;; example usage:

(def training-data (data/load-data (:train data/file-paths)))

(def test-data (data/load-data (:test data/file-paths)))

(def survivor-table (statistics/frequency-table :count [:Sex :Survived] training-data))

(def survivor-map (statistics/frequency-map :count [:Sex :Survived] training-data))

;; relative risk shows males are 2.97 times more likely to
;; die than females
(def fatalities-by-sex-fractions (statistics/fatalities-by-sex training-data))