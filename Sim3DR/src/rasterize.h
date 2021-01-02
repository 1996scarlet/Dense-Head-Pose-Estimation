#include <cmath>
#include <algorithm>

using namespace std;

class Point
{
public:
    float x, y;

public:
    Point() : x(0.f), y(0.f) {}
    Point(float x_, float y_) : x(x_), y(y_) {}

    float dot(Point p)
    {
        return this->x * p.x + this->y * p.y;
    }

    Point operator-(const Point &p)
    {
        Point np;
        np.x = this->x - p.x;
        np.y = this->y - p.y;
        return np;
    }

    Point operator+(const Point &p)
    {
        Point np;
        np.x = this->x + p.x;
        np.y = this->y + p.y;
        return np;
    }

    Point operator*(float s)
    {
        Point np;
        np.x = s * this->x;
        np.y = s * this->y;
        return np;
    }
};

void get_point_weight(float *weight, Point p, Point p0, Point p1, Point p2);

void _get_normal(float *ver_normal, float *vertices, int *triangles, int nver, int ntri);

void _rasterize(
    unsigned char *image, float *vertices, int *triangles, float *colors,
    float *depth_buffer, int ntri, int h, int w, int c, float alpha, bool reverse);
