import os
import glob

data_path = "./experiments/"

f = open(data_path + f'result.html','w')
msg = """<html>
    <head></head>
    <body>"""
figw = 40
for i in range(38):
    img = f"./{i}_gyro_split.gif"
    tmp = f"<p style=\"float: left; font-size: 11pt; text-align: center; width: {figw}%; margin-right: 1%; margin-bottom: 0.5em;\"><img src=\"{img}\" style=\"width: 100%\">Gyro Mag {i}</p>"
    msg += tmp

    img = f"./{i}_ronin_split.gif"
    tmp = f"<p style=\"float: left; font-size: 11pt; text-align: center; width: {figw}%; margin-right: 1%; margin-bottom: 0.5em;\"><img src=\"{img}\" style=\"width: 100%\">Ronin Traj {i}</p>"
    msg += tmp

    tmp = f"<p style=\"font-size: 14pt; text-align: center; margin-right: 30%;\"><b>Example {i}</b></p>"
    msg += tmp

msg += """</body>
</html>"""
f.write(msg)
f.close()

f = open(data_path + f'cluster.html','w')
msg = """<html>
    <head></head>
    <body>"""
figw = 23
for i in range(38):
    img = f"./{i}_cluster.png"
    tmp = f"<p style=\"float: left; font-size: 11pt; text-align: center; width: {figw}%; margin-right: 1%; margin-bottom: 0.5em;\"><img src=\"{img}\" style=\"width: 100%\">Day {i}</p>"
    msg += tmp

    # tmp = f"<p style=\"font-size: 14pt; text-align: center; margin-right: 30%;\"><b>Example {i}</b></p>"
    # msg += tmp

msg += """</body>
</html>"""
f.write(msg)
f.close()

f = open(data_path + f'align.html','w')
msg = """<html>
    <head></head>
    <body>"""
figw = 30
img = f"./joint.png"
tmp = f"<p style=\"float: left; font-size: 11pt; text-align: center; width: {figw*3}%; margin-right: 1%; margin-bottom: 0.5em;\"><img src=\"{img}\" style=\"width: 100%\">Draw together</p>"
msg += tmp
for i in range(38):
    img = f"./{i}_before.png"
    tmp = f"<p style=\"float: left; font-size: 11pt; text-align: center; width: {figw}%; margin-right: 1%; margin-bottom: 0.5em;\"><img src=\"{img}\" style=\"width: 100%\">Before {i}</p>"
    msg += tmp

    img = f"./{i}_after.png"
    tmp = f"<p style=\"float: left; font-size: 11pt; text-align: center; width: {figw}%; margin-right: 1%; margin-bottom: 0.5em;\"><img src=\"{img}\" style=\"width: 100%\">Single Align {i}</p>"
    msg += tmp

    img = f"./{i}_multi.png"
    tmp = f"<p style=\"float: left; font-size: 11pt; text-align: center; width: {figw}%; margin-right: 1%; margin-bottom: 0.5em;\"><img src=\"{img}\" style=\"width: 100%\">Joint Align {i}</p>"
    msg += tmp

    # tmp = f"<p style=\"font-size: 14pt; text-align: center; margin-right: 30%;\"><b>Example {i}</b></p>"
    # msg += tmp

msg += """</body>
</html>"""
f.write(msg)
f.close()

f = open(data_path + f'single-align-by-search.html','w')
msg = """<html>
    <head></head>
    <body>"""
figw = 23

filenames = sorted(glob.glob(data_path+"singlesearch/*WiFi_SfM.png"))

# tmp = "<p><b>Green: fixed reference, Red: before rotation, Blue: after rotation</b></p>"
# msg += tmp
for i in filenames:
    i = os.path.basename(i)[:-4]
    img = f"./singlesearch/{i}-flp.png"
    tmp = f"<p style=\"float: left; font-size: 11pt; text-align: center; width: {figw}%; margin-right: 1%; margin-bottom: 0.5em;\"><img src=\"{img}\" style=\"width: 100%\">{i} FLP</p>"
    msg += tmp

    img = f"./singlesearch/{i}.png"
    tmp = f"<p style=\"float: left; font-size: 11pt; text-align: center; width: {figw}%; margin-right: 1%; margin-bottom: 0.5em;\"><img src=\"{img}\" style=\"width: 100%\">Correspondence</p>"
    msg += tmp

    img = f"./singlesearch/{i}-align.png"
    tmp = f"<p style=\"float: left; font-size: 11pt; text-align: center; width: {figw}%; margin-right: 1%; margin-bottom: 0.5em;\"><img src=\"{img}\" style=\"width: 100%\">Aligned Result (no RANSAC)</p>"
    msg += tmp

    img = f"./singlesearch/{i}-align-ransac.png"
    tmp = f"<p style=\"float: left; font-size: 11pt; text-align: center; width: {figw}%; margin-right: 1%; margin-bottom: 0.5em;\"><img src=\"{img}\" style=\"width: 100%\">Aligned Result (RANSAC)</p>"
    msg += tmp

msg += """</body>
</html>"""
f.write(msg)
f.close()

f = open(data_path + f'condition-comp.html','w')
msg = """<html>
    <head></head>
    <body>"""
figw = 32

filenames = sorted(glob.glob(data_path+"corres_data/C/*WiFi_SfM.png"))

# tmp = "<p><b>Green: fixed reference, Red: before rotation, Blue: after rotation</b></p>"
# msg += tmp
for i in filenames:
    i = os.path.basename(i)[:-4]
    img = f"./corres_data/D/{i}.png"
    tmp = f"<p style=\"float: left; font-size: 11pt; text-align: center; width: {figw}%; margin-right: 1%; margin-bottom: 0.5em;\"><img src=\"{img}\" style=\"width: 100%\">{i} Diff only</p>"
    msg += tmp

    img = f"./corres_data/C/{i}.png"
    tmp = f"<p style=\"float: left; font-size: 11pt; text-align: center; width: {figw}%; margin-right: 1%; margin-bottom: 0.5em;\"><img src=\"{img}\" style=\"width: 100%\">{i} Con only</p>"
    msg += tmp
    img = f"./corres_data/DC/{i}.png"
    tmp = f"<p style=\"float: left; font-size: 11pt; text-align: center; width: {figw}%; margin-right: 1%; margin-bottom: 0.5em;\"><img src=\"{img}\" style=\"width: 100%\">{i} Diff+Con</p>"
    msg += tmp

msg += """</body>
</html>"""
f.write(msg)
f.close()

f = open(data_path + f'ransac-analysis.html','w')
msg = """<html>
    <head></head>
    <body>"""
figw = 15.0

filenames = sorted(glob.glob(data_path+"ransac/*WiFi_SfM.png"))

for i in filenames:
    i = os.path.basename(i)[:-4]

    img = f"./ransac/{i}_o.png"
    tmp = f"<p style=\"float: left; font-size: 11pt; text-align: center; width: {figw}%; margin-right: 1%; margin-bottom: 0.5em;\"><img src=\"{img}\" style=\"width: 100%\">Before</p>"
    msg += tmp

    img = f"./ransac/{i}_n.png"
    tmp = f"<p style=\"float: left; font-size: 11pt; text-align: center; width: {figw}%; margin-right: 1%; margin-bottom: 0.5em;\"><img src=\"{img}\" style=\"width: 100%\">After</p>"
    msg += tmp

    img = f"./ransac/{i}.png"
    tmp = f"<p style=\"float: left; font-size: 11pt; text-align: center; width: {figw}%; margin-right: 1%; margin-bottom: 0.5em;\"><img src=\"{img}\" style=\"width: 100%\">Put together</p>"
    msg += tmp

    img = f"./singlesearch/{i}-align.png"
    tmp = f"<p style=\"float: left; font-size: 11pt; text-align: center; width: {figw}%; margin-right: 1%; margin-bottom: 0.5em;\"><img src=\"{img}\" style=\"width: 100%\">Aligned Result (no RANSAC)</p>"
    msg += tmp

    img = f"./singlesearch/{i}-align-ransac.png"
    tmp = f"<p style=\"float: left; font-size: 11pt; text-align: center; width: {figw}%; margin-right: 1%; margin-bottom: 0.5em;\"><img src=\"{img}\" style=\"width: 100%\">Aligned Result (RANSAC)</p>"
    msg += tmp

    img = f"./singlesearch/{i}-flp.png"
    tmp = f"<p style=\"float: left; font-size: 11pt; text-align: center; width: {figw}%; margin-right: 1%; margin-bottom: 0.5em;\"><img src=\"{img}\" style=\"width: 100%\">FLP {i}</p>"
    msg += tmp

    
msg += """</body>
</html>"""
f.write(msg)
f.close()

f = open(data_path + f'pair-corres.html','w')
msg = """<html>
    <head></head>
    <body>"""
figw = 32

filenames = sorted(glob.glob(data_path+"paircorres/*.png"))

# tmp = "<p><b>Green: fixed reference, Red: before rotation, Blue: after rotation</b></p>"
# msg += tmp
for i in filenames:
    i = os.path.basename(i)[:-4]
    img = f"./paircorres/{i}.png"
    tmp = f"<p style=\"float: left; font-size: 11pt; text-align: center; width: {figw}%; margin-right: 1%; margin-bottom: 0.5em;\"><img src=\"{img}\" style=\"width: 100%\">{i}</p>"
    msg += tmp

msg += """</body>
</html>"""
f.write(msg)
f.close()