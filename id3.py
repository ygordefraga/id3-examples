import csv
import math
import json
import argparse


def load_data(filename):
    data = []
    with open(filename, "r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append(row)
    return data


def entropy(class_counts):
    total = sum(class_counts.values())
    entropy = 0
    for count in class_counts.values():
        if count != 0:
            probability = count / total
            entropy -= probability * math.log2(probability)
    return entropy


def information_gain(data, attribute, class_name):
    attribute_values = {}
    class_counts = {}
    total_count = 0

    for entry in data:
        attribute_value = entry[attribute]
        if attribute_value not in attribute_values:
            attribute_values[attribute_value] = []
        attribute_values[attribute_value].append(entry)

        class_value = entry[class_name]
        if class_value not in class_counts:
            class_counts[class_value] = 0
        class_counts[class_value] += 1
        total_count += 1

    remainder = 0
    for value_data in attribute_values.values():
        remainder += (
            len(value_data)
            / total_count
            * entropy(
                {
                    class_value: len(
                        [
                            entry
                            for entry in value_data
                            if entry[class_name] == class_value
                        ]
                    )
                    for class_value in class_counts.keys()
                }
            )
        )

    return entropy(class_counts) - remainder


def plurality_value(data, class_name):
    class_counts = {}
    for entry in data:
        class_value = entry[class_name]
        if class_value not in class_counts:
            class_counts[class_value] = 0
        class_counts[class_value] += 1
    max_class = max(class_counts, key=class_counts.get)
    return max_class, class_counts[max_class]


def decision_tree_learning(examples, attributes, class_name, parent_examples=None):
    if not examples:
        return {
            plurality_value(parent_examples, class_name)[0]: plurality_value(
                parent_examples, class_name
            )[1]
        }
    elif all(entry[class_name] == examples[0][class_name] for entry in examples):
        return {examples[0][class_name]: len(examples)}
    elif not attributes:
        return {
            plurality_value(examples, class_name)[0]: plurality_value(
                examples, class_name
            )[1]
        }
    else:
        importance_scores = {
            attribute: information_gain(examples, attribute, class_name)
            for attribute in attributes
        }
        best_attribute = max(importance_scores, key=importance_scores.get)
        tree = {}
        for value in set(entry[best_attribute] for entry in examples):
            value_examples = [
                entry for entry in examples if entry[best_attribute] == value
            ]
            subtree = decision_tree_learning(
                value_examples,
                [attr for attr in attributes if attr != best_attribute],
                class_name,
                examples,
            )
            tree[value] = subtree
        return {best_attribute: tree}


def print_json(tree):
    print(json.dumps(tree, indent=4))


def count_int_values(tree, counts={}):
    for key, value in tree.items():
        if isinstance(value, int):
            counts[key] = counts.get(key, 0) + value
        elif isinstance(value, dict):
            # Recursively count integer values in nested dictionaries
            count_int_values(value, counts)
    return counts


def predict_class(decision_tree, instance):
    key = next(iter(decision_tree))
    if isinstance(decision_tree[key], int):
        return key
    attribute_value = instance.get(key)
    subtree = decision_tree[key].get(attribute_value)
    if subtree is None:
        counts = count_int_values(decision_tree)
        return max(counts, key=counts.get)
    return predict_class(subtree, instance)


def predict_classes(decision_tree, instances):
    predictions = []
    for instance in instances:
        predicted_class = predict_class(decision_tree, instance)
        predictions.append(predicted_class)
    return predictions


def discretize_temp(temp):
    if temp <= 60:
        return "<=60"
    elif 61 <= temp <= 70:
        return "61-70"
    elif 71 <= temp <= 80:
        return "71-80"
    elif 81 <= temp <= 90:
        return "81-90"
    else:
        return ">=91"


def discretize_humidity(humidity):
    if humidity <= 70:
        return "<=70"
    elif 71 <= humidity <= 80:
        return "71-80"
    else:
        return ">=81"


if __name__ == "__main__":
    # Define the command-line arguments
    parser = argparse.ArgumentParser(description="Decision Tree Learning")
    parser.add_argument(
        "--problem",
        choices=["restaurants", "weather"],
        help="Problem to solve",
        default="restaurants",
    )
    parser.add_argument(
        "--predict-file",
        help="File containing instances to predict (optional)",
        default=None,
    )

    # Parse the command-line arguments
    args = parser.parse_args()

    # Load data based on the specified problem
    filename = (
        "data/restaurant.csv" if args.problem == "restaurants" else "data/weather.csv"
    )
    class_name = "Class" if args.problem == "restaurants" else "Play"
    data = load_data(filename)

    attributes = list(data[0].keys())
    attributes.remove(class_name)
    attributes.remove("ID")

    if args.problem == "weather":
        for entry in data:
            # Pré-processar 'Temp' e 'Humidity' para agrupá-los em blocos
            entry["Temp"] = discretize_temp(int(entry["Temp"]))
            entry["Humidity"] = discretize_humidity(int(entry["Humidity"]))

    # Executing the decision tree learning algorithm
    decision_tree = decision_tree_learning(data, attributes, class_name)

    # Printing the decision tree in JSON format
    print_json(decision_tree)

    # Predicting the class for each new instance if a prediction file is provided
    if args.predict_file:
        new_instances = load_data(args.predict_file)
        if args.problem == "weather":
            for entry in new_instances:
                # Pré-processar 'Temp' e 'Humidity' para agrupá-los em blocos
                entry["Temp"] = discretize_temp(int(entry["Temp"]))
                entry["Humidity"] = discretize_humidity(int(entry["Humidity"]))
        predictions = predict_classes(decision_tree, new_instances)

        # Displaying the predictions
        for instance, prediction in zip(new_instances, predictions):
            print(f"Instance: {instance}, Predicted Class: {prediction}")
