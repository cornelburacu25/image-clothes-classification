import os
import json
import shutil

categories = {
    'pants': ['trousers', 'jeans', 'slacks','pants'],
    't-shirt': ['tee', 'tee shirt', 'tshirt', 't-shirt', 't-shirts' , 'tshirts'],
    'skirt': ['mini', 'midi', 'maxi' , 'skirt'],
    'dress': ['gown', 'frock', 'shift', 'dress'],
    'shorts': ['bermuda', 'cargo', 'culottes', 'shorts' , 'capris'],
    'shoes': ['footwear', 'sneakers', 'boots','shoes' , 'shoe'],
    'hat': ['cap', 'beanie', 'fedora' , 'hat'],
    'longsleeve': ['sweater', 'jumper', 'cardigan', 'longsleeve'],
    'outwear': ['jacket', 'coat', 'blazer','outwear'],
    'shirt': ['blouse', 'top', 'polo','shirt', 'shirts']
}

image_folder = 'fashion-dataset/images'
style_folder = 'fashion-dataset/styles'


for category in categories:
    output_dir = os.path.join(image_folder, category)
    os.makedirs(output_dir, exist_ok=True)

for name in os.listdir(image_folder):
    if name.endswith(('.jpg', '.png','.jpeg')):
        name_json = os.path.splitext(name)[0] + '.json'
        path_json = os.path.join(style_folder, name_json)

        with open(path_json) as file:
            data = json.load(file)

        category_found = False
        for category_name, category_synonyms in categories.items():
            if any(synonym in str(data['data']['subCategory']['typeName']).lower() for synonym in category_synonyms):
                    category = category_name
                    category_found = True
                    break
            if any(synonym in str(data['data']['masterCategory']['typeName']).lower() for synonym in category_synonyms):
                category = category_name
                category_found = True
                break
            if any(synonym in str(data['data']['articleType']['typeName']).lower() for synonym in category_synonyms):
                category = category_name
                category_found = True
                break
            if category_found:
                break

        if not category_found:
            print(f'Could not find category for image file: {name}')
            continue

        input_path = os.path.join(image_folder, name)
        output_path = os.path.join(image_folder, category, name)
        if category_found:
            shutil.move(input_path, output_path)