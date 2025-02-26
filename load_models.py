import os
import shutil
import objaverse
import objaverse.xl as oxl
import trimesh
import argparse

def download_models(output_dir, models_list_file):
    if not os.path.exists(models_list_file):
        print(f"Error: {models_list_file} does not exist.")
        return

    with open(models_list_file, 'r') as f:
        uids = [line.strip() for line in f if line.strip()]

    if not uids:
        print("Error: No UIDs found in the file.")
        return

    for uid in uids:
        output_path = os.path.join(output_dir, uid)

        objects = objaverse.load_objects(uids=[uid])
        annotations = objaverse.load_annotations(uids=[uid])

        if not os.path.exists(output_path) and len(objects) > 0:
            os.makedirs(output_path, exist_ok=True)

            try:
                loaded = trimesh.load(list(objects.values())[0])
                mesh_trimesh = trimesh.util.concatenate(loaded.dump())
                # mesh_trimesh.show()
                obj_file_path = os.path.join(output_path, f"{uid}.obj")
                mesh_trimesh.export(obj_file_path, file_type='obj')
                print(f"OBJ file saved to: {obj_file_path}")
            except Exception as e:
                print(f"Failed to export {uid}.obj: {e}")
                # Delete the folder if an error occurs
                shutil.rmtree(output_path)
                print(f"Deleted folder due to error: {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', default='./assets/Models/', help='Output directory where OBJ files will be saved')
    parser.add_argument('--models_list_file', default='models_list.txt', help='Path to the text file containing UIDs')
    args = parser.parse_args()

    download_models(args.output_dir, args.models_list_file)