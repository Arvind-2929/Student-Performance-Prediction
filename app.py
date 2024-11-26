{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "44dd0a75-71d7-4dc0-aee8-49fd339ac50d",
   "metadata": {},
   "outputs": [],
   "source": [
    "from flask import Flask, jsonify, request, send_file\n",
    "import pandas as pd\n",
    "import seaborn as sb\n",
    "import matplotlib.pyplot as plt\n",
    "import numpy as np\n",
    "import sklearn.tree as tr\n",
    "import sklearn.ensemble as es\n",
    "import sklearn.linear_model as lm\n",
    "import sklearn.neural_network as nn\n",
    "import sklearn.metrics as m\n",
    "import sklearn.preprocessing as pp\n",
    "import os\n",
    "\n",
    "app = Flask(__name__)\n",
    "\n",
    "# Load the dataset\n",
    "data = pd.read_csv(\"/AI-Data.csv\")\n",
    "\n",
    "# Models dictionary\n",
    "models = {}\n",
    "\n",
    "@app.route('/graphs/<graph_type>', methods=['GET'])\n",
    "def generate_graph(graph_type):\n",
    "    \"\"\"\n",
    "    Generate graphs based on the type requested.\n",
    "    \"\"\"\n",
    "    try:\n",
    "        graph_map = {\n",
    "            \"count\": lambda: sb.countplot(x='Class', data=data, order=['L', 'M', 'H']),\n",
    "            \"semester\": lambda: sb.countplot(x='Semester', hue='Class', data=data, hue_order=['L', 'M', 'H']),\n",
    "            \"gender\": lambda: sb.countplot(x='gender', hue='Class', data=data, order=['M', 'F'], hue_order=['L', 'M', 'H']),\n",
    "            \"nationality\": lambda: sb.countplot(x='NationalITy', hue='Class', data=data, hue_order=['L', 'M', 'H']),\n",
    "        }\n",
    "\n",
    "        if graph_type not in graph_map:\n",
    "            return jsonify({\"error\": \"Invalid graph type\"}), 400\n",
    "\n",
    "        # Generate the graph\n",
    "        plt.figure(figsize=(10, 6))\n",
    "        graph_map[graph_type]()\n",
    "        plt.savefig(f\"{graph_type}.png\")\n",
    "        plt.close()\n",
    "\n",
    "        # Return the graph as a file\n",
    "        return send_file(f\"{graph_type}.png\", mimetype='image/png')\n",
    "\n",
    "    except Exception as e:\n",
    "        return jsonify({\"error\": str(e)}), 500\n",
    "\n",
    "\n",
    "@app.route('/train', methods=['POST'])\n",
    "def train_models():\n",
    "    \"\"\"\n",
    "    Train models on the dataset and return accuracies.\n",
    "    \"\"\"\n",
    "    global models\n",
    "\n",
    "    try:\n",
    "        # Data preprocessing\n",
    "        label_encoder = pp.LabelEncoder()\n",
    "        for column in data.columns:\n",
    "            if data[column].dtype == 'object':\n",
    "                data[column] = label_encoder.fit_transform(data[column])\n",
    "\n",
    "        ind = int(len(data) * 0.7)\n",
    "        features = data.values[:, :-1]\n",
    "        labels = data.values[:, -1]\n",
    "        feats_train, feats_test = features[:ind], features[ind:]\n",
    "        labels_train, labels_test = labels[:ind], labels[ind:]\n",
    "\n",
    "        # Train models\n",
    "        models['decision_tree'] = tr.DecisionTreeClassifier()\n",
    "        models['random_forest'] = es.RandomForestClassifier()\n",
    "        models['perceptron'] = lm.Perceptron()\n",
    "        models['logistic_regression'] = lm.LogisticRegression()\n",
    "        models['mlp'] = nn.MLPClassifier(activation=\"logistic\")\n",
    "\n",
    "        accuracies = {}\n",
    "        for name, model in models.items():\n",
    "            model.fit(feats_train, labels_train)\n",
    "            predictions = model.predict(feats_test)\n",
    "            accuracy = m.accuracy_score(labels_test, predictions)\n",
    "            accuracies[name] = round(accuracy, 3)\n",
    "\n",
    "        return jsonify({\"accuracies\": accuracies})\n",
    "\n",
    "    except Exception as e:\n",
    "        return jsonify({\"error\": str(e)}), 500\n",
    "\n",
    "\n",
    "@app.route('/predict', methods=['POST'])\n",
    "def predict():\n",
    "    \"\"\"\n",
    "    Use trained models to predict outcomes for given input.\n",
    "    \"\"\"\n",
    "    try:\n",
    "        if not models:\n",
    "            return jsonify({\"error\": \"Models are not trained. Train them first using /train.\"}), 400\n",
    "\n",
    "        # Get input data from the request\n",
    "        input_data = request.json\n",
    "        if not input_data:\n",
    "            return jsonify({\"error\": \"No input data provided\"}), 400\n",
    "\n",
    "        input_array = np.array(list(input_data.values())).reshape(1, -1)\n",
    "\n",
    "        predictions = {}\n",
    "        for name, model in models.items():\n",
    "            prediction = model.predict(input_array)[0]\n",
    "            predictions[name] = prediction\n",
    "\n",
    "        return jsonify({\"predictions\": predictions})\n",
    "\n",
    "    except Exception as e:\n",
    "        return jsonify({\"error\": str(e)}), 500\n",
    "\n",
    "\n",
    "@app.route('/')\n",
    "def home():\n",
    "    \"\"\"\n",
    "    Home endpoint to provide API details.\n",
    "    \"\"\"\n",
    "    return jsonify({\n",
    "        \"message\": \"Welcome to the Flask API for Data Analysis and Machine Learning.\",\n",
    "        \"endpoints\": {\n",
    "            \"/graphs/<type>\": \"Generate specific graphs (types: count, semester, gender, nationality, etc.)\",\n",
    "            \"/train\": \"Train models on the dataset.\",\n",
    "            \"/predict\": \"Predict outcomes using trained models (POST with JSON input).\"\n",
    "        }\n",
    "    })\n",
    "\n",
    "\n",
    "if __name__ == '__main__':\n",
    "    # Run the app\n",
    "    app.run(debug=True, host='0.0.0.0', port=5000)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b65bc469-1abb-48d2-8fa2-28a390aa0037",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": ""
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
