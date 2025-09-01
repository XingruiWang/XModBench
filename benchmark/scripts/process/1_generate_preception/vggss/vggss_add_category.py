import json
from collections import Counter
import matplotlib.pyplot as plt

# Load the VGGSound dataset
vggss_json = '/home/xwang378/scratch/2025/AudioBench/benchmark/Data/vggss_meta_json/vggss.json'

with open(vggss_json, 'r') as f:
    data = json.load(f)

print(f"Total number of samples: {len(data)}")

# Extract all classes
classes = [item['class'] for item in data]

# Count occurrences of each class
class_counts = Counter(classes)

print(f"\nTotal number of unique classes: {len(class_counts)}")
  
# Define simplified categories
def categorize_classes_simple(class_counts):
    categories = {
        'Musical Instruments': [],
        'Human Speech': [],
        'Human Activities': [],
        'Animal Sounds': [],
        'Transportation': [],
        'Urban Sounds': [],
        'Tools & Machinery': [],
        'Natural Sounds': [],
        'Other': []
    }
    
    for class_name, count in class_counts.items():
        assigned = False
        
        # Musical Instruments
        if ('playing' in class_name and "table tennis" not in class_name) or class_name in ['beat boxing', 'singing bowl', 'orchestra', 'disc scratching', 'tapping guitar']:
            categories['Musical Instruments'].append((class_name, count))
        
        # Human Speech
        elif (any(word in class_name for word in ['speech', 'speaking', 'singing', 'yodelling']) and "bird" not in class_name) or \
             class_name in ['singing choir']:
            categories['Human Speech'].append((class_name, count))
        
        # Human Activities
        elif any(word in class_name for word in ['people', 'baby', 'child', 'children']) or \
             class_name in ['tap dancing', 'typing on computer keyboard', 'typing on typewriter', 
                           'skateboarding', 'lip smacking', 'playing table tennis']:
            categories['Human Activities'].append((class_name, count))
        
        # Animal Sounds
        elif (any(word in class_name for word in ['dog', 'cat', 'bird', 'calling', 'chirping', 'tweeting', 
                                               'cow', 'chicken', 'turkey', 'goat', 'sheep', 'horse', 
                                               'elephant', 'lion', 'bull', 'elk', 'coyote', 'gibbon',
                                               'snake', 'alligator', 'crocodile', 'fox', 'owl', 'eagle',
                                               'penguin', 'chipmunk', 'cheetah', 'donkey', 'whale',
                                               'mouse', 'fly', 'cricket', 'otter', 'parrot']) and "plane" not in class_name and "bell" not in class_name) or \
             'barking' in class_name or 'howling' in class_name or 'hissing' in class_name or \
             'roaring' in class_name or 'bellowing' in class_name or 'braying' in class_name or \
             'bleating' in class_name or 'meowing' in class_name or 'purring' in class_name or \
             'clucking' in class_name or 'crowing' in class_name or 'gobbling' in class_name or \
             'cooing' in class_name or ('hooting' in class_name and "gun" not in class_name) or 'cawing' in class_name or \
             'trumpeting' in class_name or 'squawking' in class_name or 'buzzing' in class_name or \
             'squeaking' in class_name or 'rattling' in class_name or 'caterwauling' in class_name or \
             'growling' in class_name or 'whimpering' in class_name or 'pant-hooting' in class_name or \
             class_name in ['dinosaurs bellowing', 'cattle mooing', 
                           'woodpecker pecking tree', 'mynah bird singing', 'francolin calling',
                           'wood thrush calling', 'baltimore oriole calling', 'black capped chickadee calling',
                           'magpie calling', 'canary calling', 'cuckoo bird calling', 'barn swallow calling',
                           'warbler chirping', 'bird wings flapping', 'sea lion barking', 'chinchilla barking',
                           'chimpanzee pant-hooting', 'horse clip-clop']:
            categories['Animal Sounds'].append((class_name, count))
        
        # Transportation
        elif any(word in class_name for word in ['car', 'train', 'airplane', 'helicopter', 'motorcycle', 
                                               'bus', 'boat', 'subway', 'metro', 'underground', 'engine',
                                               'driving', 'plane']) or \
             class_name in ['skidding', 'opening or closing car electric windows', 'race car, auto racing',
                           'motorboat, speedboat acceleration', 'railroad car, train wagon', 'rowboat, canoe, kayak rowing']:
            categories['Transportation'].append((class_name, count))
        
        # Urban Sounds
        elif (any(word in class_name for word in ['siren', 'alarm', 'telephone', 'bell', 'horn', 'beep']) and "cow" not in class_name) or \
             class_name in ['slot machine', 'fireworks banging', 'missile launch',
                           'ice cream truck, ice cream van', 'reversing beeps', 'smoke detector beeping',
                           'civil defense siren', 'lighting firecrackers',
                           'church bell ringing', 'wind chime', 'air conditioning noise']:
            categories['Urban Sounds'].append((class_name, count))
        
        # Tools & Machinery
        elif any(word in class_name for word in ['electric', 'vacuum', 'blender', 'tractor', 'chainsaw',
                                               'lawn', 'lathe', 'sewing', 'grinder', 'hair dryer',
                                               'trimmer', 'blowtorch', 'sharpen', "gun shooting"]) or \
             class_name in ['using sewing machines', 'toilet flushing', 'bathroom ventilation fan running',
                           'hedge trimmer running', 'chainsawing trees', 'lawn mowing', 'forging swords',
                           'opening or closing drawers', 'tractor digging', 'popping popcorn',
                           'electric shaver, electric razor shaving']:
            categories['Tools & Machinery'].append((class_name, count))
        
        # Natural Sounds
        elif any(word in class_name for word in ['water', 'ocean', 'waterfall', 'wind']) or \
             class_name in ['squishing water', 'splashing water']:
            categories['Natural Sounds'].append((class_name, count))
        
        # Everything else goes to Other
        else:
            categories['Other'].append((class_name, count))
    
    return categories

# Categorize and display results
categorized = categorize_classes_simple(class_counts)

print("VGGSound Dataset - Simplified Class Categorization")
print("=" * 55)
print()

total_samples = sum(class_counts.values())
category_stats = []
class_to_category = {}

for category, classes in categorized.items():
    
    if classes:  # Only show categories that have classes
        
        classes.sort(key=lambda x: x[1], reverse=True)  # Sort by count
        category_samples = sum(count for _, count in classes)
        percentage = (category_samples / total_samples) * 100
        category_stats.append((category, len(classes), category_samples, percentage))
        
        print(f"{category.upper()}")
        print("-" * len(category))
        print(f"Classes: {len(classes)}, Total samples: {category_samples} ({percentage:.1f}%)")
        print()
        
        # Show top classes in each category
        for class_name, count in classes[:10]:  # Show top 10
            print(f"  {class_name:<45} {count:>3}")
        
        if len(classes) > 10:
            remaining_samples = sum(count for _, count in classes[10:])
            print(f"  ... and {len(classes)-10} more classes ({remaining_samples} samples)")
        print()
        
        for class_name in classes:
            class_to_category[class_name[0]] = category

# Summary
print("CATEGORY SUMMARY")
print("-" * 16)
category_stats.sort(key=lambda x: x[2], reverse=True)  # Sort by sample count

for category, num_classes, samples, percentage in category_stats:
    print(f"{category:<20} {num_classes:>3} classes  {samples:>5} samples ({percentage:>5.1f}%)")

print(f"\nTotal: {len(class_counts)} classes, {total_samples} samples")


import json

# Assuming you have already loaded your data and built class_to_category
# vggss_json = '/home/xwang378/scratch/2025/AudioBench/benchmark/Data/vggss.json'
# with open(vggss_json, 'r') as f:
#     data = json.load(f)

# Add categories to each instance
for instance in data:
    data_class = instance["class"]
    data_category = class_to_category[data_class]
    instance['category'] = data_category
    
print(categorized)

# Save the updated data
output_file = '/home/xwang378/scratch/2025/AudioBench/benchmark/Data/vggss_meta_json/vggss_extend_category.json'
with open(output_file, 'w') as f:
    json.dump(data, f, indent=2)

print(f"Successfully saved {len(data)} instances with categories to {output_file}")

    