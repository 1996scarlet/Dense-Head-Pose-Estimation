#include <math.h>
#include <stdlib.h>

#define max(x, y) (((x) > (y)) ? (x) : (y))
#define min(x, y) (((x) < (y)) ? (x) : (y))

struct Point{
    float x;
    float y;
};

/*
 * Directly get normal of vertices, which can be regraded as a combination of _get_tri_normal and _get_ver_normal
 */
void _get_normal(float *ver_normal, float *vertices, int *triangles, int nver, int ntri)
{
    int tri_p0_ind, tri_p1_ind, tri_p2_ind;
    float v1x, v1y, v1z, v2x, v2y, v2z;

    float *tri_normal = (float *)malloc(3 * ntri * sizeof(float));

    for (int i = 0; i < ntri; i++)
    {
        tri_p0_ind = triangles[3 * i];
        tri_p1_ind = triangles[3 * i + 1];
        tri_p2_ind = triangles[3 * i + 2];

        // counter clockwise order
        v1x = vertices[3 * tri_p1_ind] - vertices[3 * tri_p0_ind];
        v1y = vertices[3 * tri_p1_ind + 1] - vertices[3 * tri_p0_ind + 1];
        v1z = vertices[3 * tri_p1_ind + 2] - vertices[3 * tri_p0_ind + 2];

        v2x = vertices[3 * tri_p2_ind] - vertices[3 * tri_p0_ind];
        v2y = vertices[3 * tri_p2_ind + 1] - vertices[3 * tri_p0_ind + 1];
        v2z = vertices[3 * tri_p2_ind + 2] - vertices[3 * tri_p0_ind + 2];

        tri_normal[3 * i] = v1y * v2z - v1z * v2y;
        tri_normal[3 * i + 1] = v1z * v2x - v1x * v2z;
        tri_normal[3 * i + 2] = v1x * v2y - v1y * v2x;
    }

    // get ver_normal
    for (int i = 0; i < ntri; ++i)
    {
        tri_p0_ind = triangles[3 * i];
        tri_p1_ind = triangles[3 * i + 1];
        tri_p2_ind = triangles[3 * i + 2];

        for (int j = 0; j < 3; j++)
        {
            ver_normal[3 * tri_p0_ind + j] += tri_normal[3 * i + j];
            ver_normal[3 * tri_p1_ind + j] += tri_normal[3 * i + j];
            ver_normal[3 * tri_p2_ind + j] += tri_normal[3 * i + j];
        }
    }

    // normalizing
    float nx, ny, nz, det;
    for (int i = 0; i < nver; ++i)
    {
        nx = ver_normal[3 * i];
        ny = ver_normal[3 * i + 1];
        nz = ver_normal[3 * i + 2];

        det = sqrt(nx * nx + ny * ny + nz * nz);
        if (det <= 0)
            det = 1e-6;
        ver_normal[3 * i] = nx / det;
        ver_normal[3 * i + 1] = ny / det;
        ver_normal[3 * i + 2] = nz / det;
    }

    free(tri_normal);
}

// rasterization by Z-Buffer with optimization
// Complexity: < ntri * h * w * c
void _rasterize(
    unsigned char *image, float *vertices, int *triangles, float *colors,
    int ntri, int h, int w, int c, float alpha)
{
    int x, y, k;
    int tri_p0_ind, tri_p1_ind, tri_p2_ind;
    struct Point p0, p1, p2, v0, v1, v2;
    int x_min, x_max, y_min, y_max;
    float p_depth, p0_depth, p1_depth, p2_depth;
    float p_color, p0_color, p1_color, p2_color;
    float weight[3];

    float *depth_buffer = (float *)malloc(h * w * sizeof(float));
    ;

    for (int i = 0; i < ntri; ++i)
    {
        tri_p0_ind = triangles[3 * i];
        tri_p1_ind = triangles[3 * i + 1];
        tri_p2_ind = triangles[3 * i + 2];

        p0.x = vertices[3 * tri_p0_ind];
        p0.y = vertices[3 * tri_p0_ind + 1];
        p0_depth = vertices[3 * tri_p0_ind + 2];
        p1.x = vertices[3 * tri_p1_ind];
        p1.y = vertices[3 * tri_p1_ind + 1];
        p1_depth = vertices[3 * tri_p1_ind + 2];
        p2.x = vertices[3 * tri_p2_ind];
        p2.y = vertices[3 * tri_p2_ind + 1];
        p2_depth = vertices[3 * tri_p2_ind + 2];

        x_min = max((int)ceil(min(p0.x, min(p1.x, p2.x))), 0);
        x_max = min((int)floor(max(p0.x, max(p1.x, p2.x))), w - 1);

        y_min = max((int)ceil(min(p0.y, min(p1.y, p2.y))), 0);
        y_max = min((int)floor(max(p0.y, max(p1.y, p2.y))), h - 1);

        if (x_max < x_min || y_max < y_min)
        {
            continue;
        }

        for (y = y_min; y <= y_max; y++)
        {
            for (x = x_min; x <= x_max; x++)
            {
                v0.x = p2.x - p0.x;
                v0.y = p2.y - p0.y;
                v1.x = p1.x - p0.x;
                v1.y = p1.y - p0.y;
                v2.x = x - p0.x;
                v2.y = y - p0.y;

                // dot products
                float dot00 = v0.x * v0.x + v0.y * v0.y; //np.dot(v0.T, v0)
                float dot01 = v0.x * v1.x + v0.y * v1.y; //np.dot(v0.T, v1)
                float dot02 = v0.x * v2.x + v0.y * v2.y; //np.dot(v0.T, v2)
                float dot11 = v1.x * v1.x + v1.y * v1.y; //np.dot(v1.T, v1)
                float dot12 = v1.x * v2.x + v1.y * v2.y; //np.dot(v1.T, v2)

                // barycentric coordinates
                float inverDeno;
                if (dot00 * dot11 - dot01 * dot01 == 0)
                    inverDeno = 0;
                else
                    inverDeno = 1 / (dot00 * dot11 - dot01 * dot01);

                float u = (dot11 * dot02 - dot01 * dot12) * inverDeno;
                float v = (dot00 * dot12 - dot01 * dot02) * inverDeno;

                // weight
                weight[0] = 1 - u - v;
                weight[1] = v;
                weight[2] = u;

                // and judge is_point_in_tri by below line of code
                if (weight[2] >= 0 && weight[1] >= 0 && weight[0] > 0)
                {
                    p_depth = weight[0] * p0_depth + weight[1] * p1_depth + weight[2] * p2_depth;

                    if (p_depth > depth_buffer[y * w + x])
                    {
                        for (k = 0; k < c; k++)
                        {
                            p0_color = colors[c * tri_p0_ind + k];
                            p1_color = colors[c * tri_p1_ind + k];
                            p2_color = colors[c * tri_p2_ind + k];

                            p_color = weight[0] * p0_color + weight[1] * p1_color + weight[2] * p2_color;
                            image[y * w * c + x * c + k] = (unsigned char)((1 - alpha) * image[y * w * c + x * c + k] + alpha * 255 * p_color);
                        }

                        depth_buffer[y * w + x] = p_depth;
                    }
                }
            }
        }
    }

    free(depth_buffer);
}
