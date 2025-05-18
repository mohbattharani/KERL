
import argparse
from core.demo import DEMO
from pprint import pprint

'''

This is a demo script for the KERL model. It allows you to test the model with different queries. 


--model_path: Path to the model directory which is Phi-3-mini-128k-instruct
--adapter: Path to the adapter directory which is checkpoints/ - All the adapters should be in this directory. 
--max_length: Maximum length of the input sequence. Default is 1024.
--device_map: Device map for the model. Default is 'cuda:0'.

'''


model_path='/data/mohbat/models/phi/Phi-3-mini-128k-instruct/'


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default='/data/mohbat/models/phi/Phi-3-mini-128k-instruct/')
    parser.add_argument("--adapter", type=str, default='checkpoints/')
    parser.add_argument("--max_length", type=int, default=1024)
    
    parser.add_argument("--device_map", type=str, default='cuda:0')
    
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=int, default=0.05)
    parser.add_argument("--lora_bias", type=str, default="none")
  
args = parser.parse_args()
  
demo = DEMO(args)


# test for keyword extraction

q = "Which gluten-free dishes contain red onions, almonds, apples, sweet rice flour, granulated sugar and avoid blueberries, ground mace, vanilla ice cream, and meet sugars per 100g less than 11.11, total carbohydrates less than 29.57?",
q = "Can you list the healthy recipes with mayonnaise, bananas, white vinegar, white wine, dry white wine but without dried apricot halves, unsweetened chocolate, steamed rice, containing salt per 100g no more than 0.24, fiber within range (3.14, 5.82), saturated fat less than 2.49?",
q = 'Could you suggest some healthy recipes that include mayonnaise, bananas, white vinegar, and dry white wine, but exclude dried apricot halves, unsweetened chocolate, and steamed rice? The recipes should have no more than 0.24g of salt per 100g, fiber between 3.14g and 5.82g, and less than 2.49g of saturated fat.'        
print ("="*40)

output = demo.get_answer (q)
pprint (output)

'''
output = demo.extract_keywords (q)
subgraph = self.get_subgraph (keywords)
print ("Extracted keywords:")
print(output)
print ("="*40)

dishes = demo.get_recommendation (q, subgraph)
print ("Recommended dishes:")
print(dishes)
print ("="*40)

nutrition = demo.get_nutrition (dishes[0])
print (f"Nutrition for the recommended dish: {dishes[0]}")
print (nutrition)
print ("="*40)

recipe = demo.get_recipe(dishes[0])
print (f"Recipe for the recommended dish: {dishes[0]}")
print (recipe)

'''