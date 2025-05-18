import json, random
from pprint import pprint


random.seed(0)

prompts_step1 = [
    "Extract the tags and ingredients mentioned in the question provided.",
    "Identify all tags and list any ingredients from the following question.",
    "Please analyze the question below and extract relevant tags and ingredients.",
    "From the question provided, determine the associated tag and ingredients.",
    "Scan the question and list any identified tag and ingredients.",
    "Identify and extract tag along with ingredients in this question.",
    "Analyze the question to find tag and ingredients references.",
    "Identify tags and list ingredients based on the provided question.",
    "Please highlight any tags and ingredients mentioned in this question.",
    "From the given question, extract both tags and ingredient names.",
    "Examine the question and retrieve associated tag and ingredients.",
    "Please find the tags and ingredients present in the provided question.",
    "Extract relevant tags and ingredients from this question.",
    "Identify tags and ingredients that appear in the question.",
    "List tag and ingredients found within the given question.",
    "Identify any tag and ingredients in the question provided.",
    "Please provide a list of tags and ingredients from these question.",
    "Analyze the question and determine the tags and ingredients present.",
    "Identify all tags and ingredients mentioned in these question.",
    "Extract tags and list ingredients associated with the following question."
]

prompts_step2 = 'Your response should contain only names of recommended recipes from context. If you do not know answer, just return an empty string.'

def convert_recipe_to_str (recipe):
    title = "Title:" + recipe['title']
    ingredients = "Ingredients:," + ", ".join(recipe['ingredients'])
    nutrition = "Nutritions:," + ", ".join([f"{k}:{v}" for k,v in recipe['nutritions'].items()])
    nutrition = nutrition.replace ("recipe", "")
    nutrition = nutrition.replace ("_", " ") 
    
    return "\n".join([title, ingredients, nutrition])

class KGQA ():
    def __init__(self, dir, partition = 'train', context_size = 3) -> None:
        self.dir = dir
        self.qas = json.load(open(f"{dir}/{partition}_constraint_qa.json", 'r'))
        self.data = json.load(open(f"{dir}/kg_dict.json", 'r'))
        self.context_size = context_size

    def __len__(self):
        return len(self.qas)
    
    def generate_test_sample (self, idx = 0):
        q = self.qas[idx]
        graphs = self.data[q['tag']]
        
        sample = {}
        sample = {'q': q['q'], 
                  'a': q['a'], 
                  'context': graphs['graphs']
                  }

        return sample

    def generate_train_sample_step1 (self, idx = -1):
        q = self.qas[idx]    
        ingredients = ", ".join (q['have_ingredients']) + ", " + ", ".join(q['must_not_have_ingredients'])
            
        prompt = random.choice(prompts_step1)
        q = f"Question: {q['q']}. {prompt}"
        a = "Tag: " + self.qas[idx]['tag'] + "\n" + f"Ingredients: {ingredients}"
        
        sample = {'q': q, 'a': a }

        return sample
    
    def generate_test_sample (self, idx, context_size = 3):
        q = self.qas[idx]
        graphs = self.data[q['tag']]
        if (context_size == -1):
            context_size = len(q['a'])
            
        negative_indices = [i for i in range(len(graphs)) if graphs[i]['title'] not in q['a']]   
        random.shuffle(negative_indices) 
        negative_indices = negative_indices[:context_size] 
        positive_indices = q['a']   
        random.shuffle (positive_indices)
        positive_indices = positive_indices[:context_size] #random.sample(q['a'], min (context_size, len(q['a'])))
        answer = [graphs['graphs'][i]['title'] for i in positive_indices]
        
        context_indices = negative_indices + positive_indices
        
        random.shuffle(context_indices) 
        context = [graphs['graphs'][i] for i in context_indices]    
        
        sample = {}
        sample = {'q': q['q'], 
                  'a': answer, 
                  'context': context
                  }

        return sample
     
    def generate_train_sample (self, idx = -1):
        q = self.qas[idx]
        graphs = self.data[q['tag']]['graphs']
        # select recipe graphs for recipes that are not in the answer
        negatives = [g for g in graphs if g['title'] not in q['a']] 
        positves = [g for g in graphs if g['title'] in q['a']]   
        # sample few negatives 
        context_size = random.randint (1, self.context_size) 
        negatives = random.sample(negatives, min ( context_size, len(negatives)))
        context_size = random.randint (1, self.context_size)
        positves = random.sample(positves, min ( context_size, len(positves)))
        
        answer = [p['title'] for p in positves]
         
        if (random.random() > 0.5):
            context = positves + negatives
        elif (random.random() > 0.5):
            context = positves
        else:
            context = negatives 
            answer = []
            
        random.shuffle(context)
        context2 = "" 
        for c in context:
            context2 += convert_recipe_to_str(c) + "\n"
        
        q = f"Question: {q['q']}. {prompts_step2} Context: {context}"
        a = "\n".join([f"{i+1}. {s}" for i, s in enumerate(answer)])
        sample = {'q': q, 'a': a }
                
        return sample

    def to_conversation (self, q, a, i = 0):
        
        #print (f"id: {i}, q: {len(q)}, a: {len(a)}")
        
        chat = { 
            "id": i,
            "conversations": []
        }

        chat ["conversations"].append(
                    {
                        "role": "user",
                        "content": q  
                    })

        chat ["conversations"].append(
                    {
                        "role": "assistant",
                        "content": a
                    })
        
        return chat
    
    def get_conversation (self, i=-1, step = -1):
        if (i == -1):
            i = random.randint(0, len(self.qas)-1)     
        if (i >= len(self.qas)):
            i = i % len(self.qas)
    
        if (step == -1):
            step = random.sample([1, 2, 2], 1)[0]
         
        if (step == 1):
            sample =  self.generate_train_sample_step1 (i)
        else:
            sample =  self.generate_train_sample (i)
            if (len(sample['a']) > 8000):
                i = i + 1
                sample =  self.generate_train_sample (i)
        
        chat = self.to_conversation (sample['q'], sample['a'], i)
        
        return chat


#kgqa = KGQA('/data/mohbat/KGQA/KGQA2/data/KGQA2/health/store/', partition='train', context_size=3)
#kgqa.get_conversation(2471)
#pprint ()