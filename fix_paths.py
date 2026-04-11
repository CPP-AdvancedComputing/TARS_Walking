import re

content = open('robot.urdf', encoding='utf-8').read()
content = content.replace('package://assets\\', '')
content = content.replace('<robot name="C:\\Users\\anike\\tars-urdf">', '<robot name="tars">')
content = re.sub(r'</link>\s*</link>', '</link>', content)

# Remove lead screw meshes - too large for MuJoCo
content = re.sub(r'<(?:visual|collision)>\s*<origin[^/]*/>\s*<geometry>\s*<mesh filename="lead_screw[^"]*"/>\s*</geometry>\s*(?:<material[^>]*>\s*<color[^/]*/>\s*</material>\s*)?</(?:visual|collision)>', '', content)

open('robot_mujoco.urdf', 'w', encoding='utf-8').write(content)

import xml.etree.ElementTree as ET
try:
    ET.parse('robot_mujoco.urdf')
    print('XML valid!')
except ET.ParseError as e:
    print('Error:', e)