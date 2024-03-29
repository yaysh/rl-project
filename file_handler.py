import json

class FileHandler:


    def __init__(self):
        self.steps = None
        self.decay_rate = None
        self.epsilon = None
        self.rewards = None
        self.epsilons = None
        self._load_data()
        

    def _load_data(self):
        try:
            with open('parameters.json') as f:
                data = json.load(f)
            json_data = json.loads(data)
            self.decay_rate = float(json_data['decay_rate'])
            self.steps = int(json_data['steps'])
            self.epsilon = float(json_data['epsilon'])
            self.epsilons = json_data['epsilons']
            self.rewards = json_data['rewards']
        except:
            print("\nJSON file is empty. No paramaters where loaded\n")
                

    def write_to_file(self, decay_rate, steps, epsilon, rewards, epsilons):
        data = json.dumps({
            'decay_rate': str(decay_rate), 
            'steps': str(steps),
            'epsilon': str(epsilon),
            'rewards': rewards,
            'epsilons': epsilons
            })
        with open("parameters.json", "w") as f:
            json.dump(data, f)

        


