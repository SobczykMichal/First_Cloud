from scipy.stats import norm
from csv import writer
def generate_points(num_points, x0,rangeX,y0, rangeY, z0,ranegZ):
    dsitribution_x = norm(loc=x0, scale=rangeX)
    dsitribution_y = norm(loc=y0, scale=rangeY)
    dsitribution_z = norm(loc=z0, scale=ranegZ)
    x = dsitribution_x.rvs(size=num_points)
    y = dsitribution_y.rvs(size=num_points)
    z = dsitribution_z.rvs(size=num_points)
    points = zip(x, y, z)
    return points

if __name__ == '__main__':
    cloud_points = generate_points(2000,0,200,0,200,0.5,0)
    with open('dane_punkty.xyz', 'w', encoding='utf-8', newline='\n') as csvfile:
        csvwriter=writer(csvfile)
        for p in cloud_points:
            csvwriter.writerow(p)