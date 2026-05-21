import subprocess
import sys
import tempfile
import textwrap
import os

def create_blender_script():
    return textwrap.dedent("""
        import bpy
        import sys
        import os
        import shutil

        def main():
            argv = sys.argv
            argv = argv[argv.index("--") + 1:]
            if len(argv) != 1:
                print("Usage: blender --background --python <script> -- <stl_path>")
                return

            input_path = os.path.abspath(argv[0])
            filename = os.path.basename(input_path)
            dirname = os.path.dirname(input_path)
            backup_dir = os.path.join(dirname, "backup")
            os.makedirs(backup_dir, exist_ok=True)

            # Load fresh scene
            bpy.ops.wm.read_factory_settings(use_empty=True)

            # Backup original
            backup_path = os.path.join(backup_dir, filename)
            shutil.copyfile(input_path, backup_path)
            print(f"[Blender] Backup saved to: {backup_path}")

            bpy.ops.wm.stl_import(filepath=input_path)
            obj = bpy.context.selected_objects[0]
            bpy.context.view_layer.objects.active = obj

            # Decimate modifier
            mod = obj.modifiers.new(name="Decimate", type='DECIMATE')
            mod.decimate_type = 'COLLAPSE'
            mod.ratio = 0.2
            bpy.ops.object.modifier_apply(modifier=mod.name)

            # Export
            bpy.ops.wm.stl_export(filepath=input_path, export_selected_objects=True)
            print(f"[Blender] Exported decimated STL to: {input_path}")

        if __name__ == "__main__":
            main()
    """)


def process_stl_files_in_directory(directory):
    blender_script = create_blender_script()

    # Write Blender script to temporary file
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as tmp:
        tmp.write(blender_script)
        blender_script_path = tmp.name

    try:
        # Walk directory and find STL files
        for root, _, files in os.walk(directory):
            for filename in files:
                if filename.lower().endswith(".stl"):
                    stl_path = os.path.join(root, filename)
                    print(f"Processing: {stl_path}")
                    try:
                        subprocess.run([
                            "blender", "--background",
                            "--python", blender_script_path,
                            "--", stl_path
                        ], check=True)
                    except subprocess.CalledProcessError as e:
                        print(f"Error processing {stl_path}: {e}")
    finally:
        os.remove(blender_script_path)

def main():
    if len(sys.argv) != 2:
        print("Usage: python batch_decimate_stl.py <directory_path>")
        sys.exit(1)

    directory = os.path.abspath(sys.argv[1])
    if not os.path.isdir(directory):
        print(f"Error: '{directory}' is not a valid directory.")
        sys.exit(1)

    process_stl_files_in_directory(directory)

if __name__ == "__main__":
    main()
