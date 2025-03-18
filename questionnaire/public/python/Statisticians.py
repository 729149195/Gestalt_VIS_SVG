import json

class LayerDataExtractor:
    def __init__(self, input_path, output_path):
        self.input_path = input_path  # Input file path
        self.output_path = output_path  # Output file path

    def process(self):
        data = self._load_data()
        layer_data = self._extract_layers(data)
        # Creating a single top-level object instead of wrapping the array in an object
        structured_data = self._create_single_top_level_object(layer_data)
        self._save_data(structured_data)

    def _load_data(self):
        # Load JSON file
        with open(self.input_path, 'r') as file:
            return json.load(file)

    def _extract_layers(self, data):
        # Extract layer data for each node
        layer_data = []
        for node_id, node_data in data.items():
            if 'layer' in node_data:
                layer_info = {
                    'id': node_id,
                    'layer': node_data['layer']
                }
                layer_data.append(layer_info)
        return layer_data

    def _create_single_top_level_object(self, data):
        # Create a single top-level object with a "name" and "children" structure
        structure = {"name": "flare", "children": []}  # Assuming top-level name is "flare"
        current_level = structure["children"]

        for item in data:
            layers = item["layer"]
            if not layers:  # If no layer info, add node directly under "flare"
                current_level.append({"name": item["id"], "value": 1})
                continue

            current_level = structure["children"]
            for layer in layers[:-1]:  # Iterate to the second last element
                found = False
                for child in current_level:
                    if child.get("name") == layer:
                        current_level = child.get("children", [])
                        found = True
                        break
                if not found:
                    new_node = {"name": layer, "children": []}
                    current_level.append(new_node)
                    current_level = new_node["children"]

            # Handle the last element, add as a node with "value"
            current_level.append({"name": item["id"], "value": 1})

        return structure

    def _save_data(self, structured_data):
        # Save the structured data to a new JSON file
        with open(self.output_path, 'w') as file:
            json.dump(structured_data, file, indent=4)


# # 示例使用（请在实际使用时取消注释）
# extractor = LayerDataExtractor('./public/python/data/extracted_nodes.json', './public/python/data/layer_data.json')
# extractor.process()