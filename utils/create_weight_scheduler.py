import math 
import matplotlib.pyplot as plt
from enum import Enum
from typing import Dict, Any
class RampMode(Enum):
    LINEAR      = "linear"
    SIGMOID     = "sigmoid"
    COSINE      = "cosine"
    EXPONENTIAL = "exponential"
    QUADRATIC   = "quadratic"
    CUBIC       = "cubic"

class WeightScheduler:
    def __init__(self, weight_max:float, weight_min:float, min_epochs:int, warmup_epochs:int, mode:RampMode=RampMode.SIGMOID):
        """
        Growth scheduler for weight parameters.
        
        Args:
            weight_max (float): Maximum weight value at the end of warm-up
            weight_min (float): Minimum weight value.
            min_epochs (int): The number of epochs the weight remains low.
            warmup_epochs (int): The number of epochs over which the weight increases from min to max.
            mode (RampMode): The ramping mode to use (linear, sigmoid, cosine, or exponential)
        """
        self.weight_max = weight_max
        self.weight_min = weight_min
        self.min_epochs = min_epochs
        self.warmup_epochs = warmup_epochs
        self.mode = mode
        self.last_epoch = 0

    def step(self):
        """
        Increment the epoch count for weight scheduling.
        """
        self.last_epoch += 1

    def get_weight(self):
        """
        Get the weight value based on the current epoch and ramp mode.    
        """
        if self.last_epoch < self.min_epochs:
            return self.weight_min
        elif self.last_epoch < self.min_epochs + self.warmup_epochs:
            progress = (self.last_epoch - self.min_epochs) / self.warmup_epochs
            
            if self.mode == RampMode.LINEAR:
                # Simple linear interpolation
                weight = self.weight_min + (self.weight_max - self.weight_min) * progress
            
            elif self.mode == RampMode.SIGMOID:
                # Smooth S-shaped curve
                weight = self.weight_min + (self.weight_max - self.weight_min) / (1 + math.exp(-10 * (progress - 0.5)))
            
            elif self.mode == RampMode.COSINE:
                # Cosine curve (smooth at both ends)
                weight = self.weight_min + (self.weight_max - self.weight_min) * (1 - math.cos(progress * math.pi)) / 2
            
            elif self.mode == RampMode.EXPONENTIAL:
                # Exponential curve (starts slow, ends fast)
                weight = self.weight_min + (self.weight_max - self.weight_min) * (math.exp(3 * progress) - 1) / (math.exp(3) - 1)
            
            elif self.mode == RampMode.QUADRATIC:
                # Quadratic curve (accelerating)
                weight = self.weight_min + (self.weight_max - self.weight_min) * progress ** 2
            
            elif self.mode == RampMode.CUBIC:
                # Cubic curve (slower start, faster end)
                weight = self.weight_min + (self.weight_max - self.weight_min) * progress ** 3
            
            return weight
        else:
            return self.weight_max

def create_weight_scheduler(config:Dict[str, Any]):
    weight_scheduler_config = config['weight_scheduler']

    if weight_scheduler_config['mode'] == None:
        return None
    elif weight_scheduler_config['mode'] == 'linear':
        mode = RampMode.LINEAR
    elif weight_scheduler_config['mode'] == 'sigmoid':
        mode = RampMode.SIGMOID
    elif weight_scheduler_config['mode'] == 'cosine':
        mode = RampMode.COSINE
    elif weight_scheduler_config['mode'] == 'exponential':
        mode = RampMode.EXPONENTIAL
    elif weight_scheduler_config['mode'] == 'quadratic':
        mode = RampMode.QUADRATIC
    elif weight_scheduler_config['mode'] == 'cubic':
        mode = RampMode.CUBIC
    else:
        raise ValueError(f"Invalid weight scheduler mode: {weight_scheduler_config['mode']}")

    return WeightScheduler(weight_scheduler_config['weight_max'],weight_scheduler_config['weight_min'], weight_scheduler_config['min_epochs'], weight_scheduler_config['warmup_epochs'], mode)

if __name__ == "__main__":
    epochs = 40
    plt.figure(figsize=(12, 6))
    
    for mode in RampMode:
        scheduler = WeightScheduler(weight_max=1.0, weight_min=0.0, min_epochs=10, warmup_epochs=20, mode=mode)
        weights = []
        for epoch in range(epochs):
            scheduler.step()
            weights.append(scheduler.get_weight())
        plt.plot(weights, linewidth=2, label=mode.value)
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Weight Value', fontsize=12)
    plt.title('Weight Schedules with Different Ramp Modes', fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.show()
