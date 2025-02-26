import argparse
import subprocess
import os
import random

random.seed(1234)

parser = argparse.ArgumentParser()
parser.add_argument("--object_folder_path", type=str, required=True, help="Path to object folder")
parser.add_argument("--material_folder_path", type=str, required=True, help="Path to material folder")
parser.add_argument("--output_folder_path", type=str, required=True, help="Path to output folder")
parser.add_argument("--num_scenes", type=int, default=10, help="Number of scenes to generate")
parser.add_argument("--blender_script", type=str, default="./blender_script.py", help="Path to blender script")
parser.add_argument("--blender_executable", type=str, default="C:/Program Files/Blender Foundation/Blender 3.5/blender", help="Path to blender executable")

def main(args):
    
    for i in range(args.num_scenes):
        output_folder_path = os.path.join(args.output_folder_path, "{:08}.data".format(i))
        if not os.path.exists(output_folder_path):
            os.makedirs(output_folder_path)
        
        # light_type 1: point, 2: sun
        light_type = random.randint(1, 2)

        command = [
            args.blender_executable,
            "-b", 
            "-P", args.blender_script,
            "--", 
            "--object_folder_path", args.object_folder_path,
            "--material_folder_path", args.material_folder_path,
            "--output_folder_path", output_folder_path,
            "--light_type", str(light_type),
        ]
    
        subprocess.run(command)
    
    
if __name__ == '__main__':
    args = parser.parse_args()

    main(args)