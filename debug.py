import mujoco

model = mujoco.MjModel.from_xml_path(r"C:\Users\anike\tars-urdf\tars_mjcf.xml")
print("Joints:", model.njnt)
print("Actuators:", model.nu)

for i in range(model.njnt):
    print(f"  joint {i}:", model.joint(i).name)
