
import torch, re, random, json
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel




class Model:
    def __init__(self, args) -> None:
        self.args = args
        self.model_path = args.model_path
        self.temperature = 0.2
        self.generation_args = {
            "temperature" :  self.temperature,
            "num_beams" : 1,
            "max_new_tokens": self.args.max_length,
            "do_sample" : True if self.temperature > 0 else False,
            'use_cache':True,
            "return_full_text": False, 
        }
        
        self.system_messge_default = {"role": "system", "content": "You are a knowledgeable language assistant with a deep understanding of food recipes. Leveraging the provided context your role is to assist the user with a variety of tasks using natural language. Your response should contain only names of recommended recipes from context. If you don't know answer, just return an empty string"}        

    
    def load_model(self):
        kwargs = {"device_map": self.args.device_map}
        # Load the base model
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path, low_cpu_mem_usage=True, trust_remote_code=True, **kwargs)
        # Load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        
        if self.args.adapter is not None:    
            self.model.load_adapter(self.args.adapter + 'keywords', adapter_name='keywords')
            self.model.load_adapter(self.args.adapter + 'rec', adapter_name='reco')
            self.model.load_adapter(self.args.adapter + 'nutri', adapter_name='nutri') 
            self.model.load_adapter(self.args.adapter + 'instruct', adapter_name='instruct')   
            
        self.pipe = pipeline( 
            "text-generation", 
            model=self.model, 
            tokenizer=self.tokenizer, 
            #device = args.device_map,
        ) 
                
        
    def load_lora_weights(self, model, lora_weights_path, kwargs):
        """
        Loads the LoRA adapter weights into the model.
        """
        # Load the LoRA weights from the file
        lora_weights = torch.load(lora_weights_path)
        
        # Apply the LoRA weights to the model parameters
        for name, param in lora_weights.items():
            if name in model.state_dict():
                model.state_dict()[name].copy_(param)
            else:
                print(f"Warning: LoRA weight {name} not found in model parameters.")

        # Optionally, you can re-apply the PeftModel functionality if needed
        #model = PeftModel(model, lora_weights_path)
        return model


    def forward_pipe (self, q):
        messages = [self.system_messge_default, {"role": "user", "content": f"{q}"}] 
        output = self.pipe(messages, **self.generation_args) 
        out = output[0]['generated_text'] 
        return out
    

class Keywords():
    def __init__(self, args) -> None:
        self.args = args
    
    def parse_tags_ingredients(self, value):
        # Parse "Tag"
        tag_match = re.search(r"Tag:\s*(.*)", value)
        tag = tag_match.group(1) if tag_match else []
        if (isinstance(tag, str)):
            tag = tag.strip()
            tag = [tag]

        # Parse "Use ingredients"
        use_ingredients_match = re.search(r"Use ingredients:\s*(.*)", value)
        use_ingredients = (
            [item.strip() for item in use_ingredients_match.group(1).split(",")]
            if use_ingredients_match
            else []
        )

        # Parse "Avoid ingredients"
        avoid_ingredients_match = re.search(r"Avoid ingredients:\s*(.*)", value)
        avoid_ingredients = (
            [item.strip() for item in avoid_ingredients_match.group(1).split(",")]
            if avoid_ingredients_match
            else []
        )

        return {'tag':tag, 'use_ingredients':use_ingredients, 'avoid_ingredients': avoid_ingredients}

    def extract_keywords (self, q, model):
        
        model.model.set_adapter('keywords')
        
        prompt =  "Extract the tags and ingredients mentioned in the question provided."
        q = f"Question: {q}. {prompt}"
        a = model.forward_pipe (q)
        a = self.parse_tags_ingredients(a)
        return a

    def forward (self, q, model):
        a = self.extract_keywords (q, model)
        return a

class Recommendation():
    def __init__(self, args) -> None:
        self.args = args
    
    def parse_dishes(self,text):
        # Using regex to extract dish names after number-dot-space pattern
        dishes = set()
        for line in text.strip().split('\n'):
            match = re.match(r'\d+\.\s*(.*)', line)
            if match:
                dish_name = match.group(1).strip()
                if dish_name:
                    dishes.add(dish_name)
        return list(dishes)

    def forward (self, q, model):    
        model.model.set_adapter('reco')
        a = model.forward_pipe (q)
        dishes = self.parse_dishes(a)
        return dishes
    

class NutritionGeneration():
    def __init__(self, args) -> None:
        self.args = args
    
    def forward (self, dish_name, model):   
        q = f"Generate the nutrition information for the dish named {dish_name}." 
        model.model.set_adapter('nutri')
        a = model.forward_pipe (q)
        return a

class RecipeGeneration():
    def __init__(self, args) -> None:
        self.args = args
    
    def forward (self, dish_name, model):   
        q = f"Generate the recipe for {dish_name}." 
        model.model.set_adapter('instruct')
        a = model.forward_pipe (q)
        return a

class DEMO ():
    def __init__(self, args) -> None:
        self.args = args
        self.model = Model(args)
        self.model.load_model()
        self.keywords = Keywords(args)
        self.recom = Recommendation(args)
        self.nutri = NutritionGeneration(args)
        self.recipe = RecipeGeneration(args)
        self.ing_dish_map = json.load (open('core/ing_dish_map.json'))
        self.graphs = json.load (open('core/kg_dict.json'))['healthy']['graphs']
        

    def save_json (self, data, file_name):
        with open(file_name, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)


    def extract_keywords (self, q):
        output = self.keywords.forward (q, self.model)
        return output
    
    def get_subgraph (self, keywords):
        
        ingrs = keywords['use_ingredients'] 
        
        dishes = set()
        all_dishes = []
        for ing in ingrs:
            if ing in self.ing_dish_map:
                all_dishes.extend(self.ing_dish_map[ing])
                
                if (not dishes):
                    dishes = set(self.ing_dish_map[ing])
                else:   
                    dishes &= set(self.ing_dish_map[ing])
        
        dishes = list(set(dishes))
        
        # If none of the dishes share the ingredients, just use all dishes that do have at least one of the ingredients
        # This is done for Demo only
        if (len(dishes) == 0):
            dishes = list(set(all_dishes))
        
        graph = []
        for g in self.graphs:
            if g['title'] in dishes:
                graph.append(g)
        
        return graph
    
    def convert_recipe_to_str (self, recipe):
        if (isinstance(recipe, str)):
            return recipe
        
        title = "Title:" + recipe['title']
        ingredients = "Ingredients:," + ", ".join(recipe['ingredients'])
        if (isinstance(recipe['nutritions'], str)):
            nutrition = recipe['nutritions']
        else:
            nutrition = "Nutritions:," + ", ".join([f"{k}:{v}" for k,v in recipe['nutritions'].items()])

        if ('instruct' in recipe):
            ingredients = ingredients +'\n' + recipe['instruct']
        
        return "\n".join([title, ingredients, nutrition])

    def get_recommendation (self, query, subgraph = []):
        if (len(subgraph) == 0):
            outputs = self.recom.forward (query, self.model)
        else:
            outputs = set()
            for i in range (0, len(subgraph), 3):
                context = subgraph[i: i+3]
                context = "\n".join ([self.convert_recipe_to_str(c) for c in context])
                q = f"Question: {query}. Context: {context}" + "Answer only names of recipes relevant to the query. "
                output = self.recom.forward (q, self.model)
                for out in output:
                    if (len(out) > 0):
                        outputs.add  (out)
            
        return list(outputs)
    
    def get_nutrition (self, dish_name):
        output = self.nutri.forward (dish_name, self.model)
        return output
    
    def get_recipe (self, dish_name):
        output = self.recipe.forward (dish_name, self.model)
        return output
    
    def get_answer (self, query, troubleshooting=False):
        keywords = self.extract_keywords (query)
        subgraph = self.get_subgraph (keywords)
        
        for i in range (3): 
            recommendation = self.get_recommendation (query, subgraph)
            if (len(recommendation) > 0):
                break
        
        select_recipe = random.choice(recommendation)
        nutri = self.get_nutrition (select_recipe)
        recipe = self.get_recipe (select_recipe)
        
        keywords.update({'recommendation recipes': recommendation})
        keywords.update({'Recipe selected': select_recipe})
        keywords.update({'subgraph': subgraph})
        keywords.update({'nutri': nutri})
        keywords.update({'recipe': recipe})
        keywords.update({'query': query})
        
        return keywords