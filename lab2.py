from punkty import generate_points
from csv import writer

if __name__ == '__main__':
    cloud_points1 = generate_points(2000, 0, 200, 0, 200, 0.5, 0)
    cloud_points2 = generate_points(3200, 100, 200, 1, 0, 72, 34)
    cloud_points3 = generate_points(4000, 100, 50, 100, 50, 70, 100)

    with open('dane_cwiczenie2.xyz', 'w', encoding='utf-8', newline='\n') as csvfile:
        csvwriter=writer(csvfile)
        for p in cloud_points1:
            csvwriter.writerow(p)
        for p in cloud_points2:
            csvwriter.writerow(p)
        for p in cloud_points3:
            csvwriter.writerow(p)
