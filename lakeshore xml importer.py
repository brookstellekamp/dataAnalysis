from brooks.LakeshoreHall import TDepHall

field = 2 #Tesla

thickness = 200 #nm

source = '/Users/mtelleka/Documents/GT/data/AlGaN Hall/temp loop hall-R520-40K-370K.hres'

RHall, VDP, mu, carriers = TDepHall(source, thickness, field)