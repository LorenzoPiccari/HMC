import numpy as np

def create_line(num_points, coef, interc, lim, rng):
    x = rng.uniform(0,1,num_points)*(lim[1]-lim[0])+lim[0]
    y = coef * x + interc
    return x,y

def generate_fake_sample( x, y, err_x, err_y, rng):
    fake_x = rng.normal(x, err_x)
    fake_y = rng.normal(y, err_y)
    return fake_x, fake_y

def get_line(num_points, coef, interc, err_x, err_y, lim, rng = np.random):
    x , y = create_line(num_points, coef, interc, lim, rng)
    fake_x, fake_y = generate_fake_sample(x, y, err_x, err_y, rng)
    return fake_x, fake_y, x, y