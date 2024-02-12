import csv
import math
import json


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


def information_gain(data, attribute):
    attribute_values = {}
    class_counts = {}
    total_count = 0

    for entry in data:
        attribute_value = entry[attribute]
        if attribute_value not in attribute_values:
            attribute_values[attribute_value] = []
        attribute_values[attribute_value].append(entry)

        class_value = entry["Class"]
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
                        [entry for entry in value_data if entry["Class"] == class_value]
                    )
                    for class_value in class_counts.keys()
                }
            )
        )

    return entropy(class_counts) - remainder


def plurality_value(data):
    class_counts = {}
    for entry in data:
        class_value = entry["Class"]
        if class_value not in class_counts:
            class_counts[class_value] = 0
        class_counts[class_value] += 1
    max_class = max(class_counts, key=class_counts.get)
    return max_class, class_counts[max_class]


def decision_tree_learning(examples, attributes, parent_examples=None):
    if not examples:
        return {
            plurality_value(parent_examples)[0]: plurality_value(parent_examples)[1]
        }
    elif all(entry["Class"] == examples[0]["Class"] for entry in examples):
        return {examples[0]["Class"]: len(examples)}
    elif not attributes:
        return {plurality_value(examples)[0]: plurality_value(examples)[1]}
    else:
        importance_scores = {
            attribute: information_gain(examples, attribute) for attribute in attributes
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
                examples,
            )
            tree[value] = subtree
        return {best_attribute: tree}


def print_json(tree):
    print(json.dumps(tree, indent=4))


# Carregando os dados do arquivo CSV
data = load_data("data/restaurant.csv")

# Obtendo os nomes dos atributos (exceto 'ID' e 'Class')
attributes = list(data[0].keys())
attributes.remove("ID")
attributes.remove("Class")

# Executando o algoritmo de aprendizado da árvore de decisão
decision_tree = decision_tree_learning(data, attributes)

# Exibindo a árvore de decisão no formato JSON
print_json(decision_tree)
