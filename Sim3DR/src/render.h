#include <math.h>
#include <stdlib.h>
#include <stdio.h>

#define max(x, y) (((x) > (y)) ? (x) : (y))
#define min(x, y) (((x) < (y)) ? (x) : (y))
#define clip(_x, _min, _max) min(max(_x, _min), _max)

struct Point
{
    float x;
    float y;
};

// rasterization by Z-Buffer with optimization
void _rasterize(unsigned char *image,
                const float *vertices,
                const int *triangles,
                const float *colors,
                int ntri, int h, int w, int c)
{
    int x, y, k, color_index;
    int tri_p0_ind, tri_p1_ind, tri_p2_ind;
    struct Point p0, p1, p2, v0, v1, v2;
    int x_min, x_max, y_min, y_max;
    float p_depth, p0_depth, p1_depth, p2_depth;
    float p_color, inverDeno, barycentric, t, u, v;
    float dot00, dot01, dot11, dot02, dot12;

    float *depth_buffer = (float *)calloc(h * w, sizeof(float));

    for (int i = 0; i < ntri; ++i)
    {
        tri_p0_ind = triangles[3 * i];
        tri_p1_ind = triangles[3 * i + 1];
        tri_p2_ind = triangles[3 * i + 2];

        p0.x = vertices[tri_p0_ind];
        p0.y = vertices[tri_p0_ind + 1];
        p0_depth = vertices[tri_p0_ind + 2];

        p1.x = vertices[tri_p1_ind];
        p1.y = vertices[tri_p1_ind + 1];
        p1_depth = vertices[tri_p1_ind + 2];

        p2.x = vertices[tri_p2_ind];
        p2.y = vertices[tri_p2_ind + 1];
        p2_depth = vertices[tri_p2_ind + 2];

        x_min = max((int)ceil(min(p0.x, min(p1.x, p2.x))), 0);
        x_max = min((int)floor(max(p0.x, max(p1.x, p2.x))), w - 1);

        y_min = max((int)ceil(min(p0.y, min(p1.y, p2.y))), 0);
        y_max = min((int)floor(max(p0.y, max(p1.y, p2.y))), h - 1);

        if (x_max < x_min || y_max < y_min)
            continue;

        v0.x = p2.x - p0.x;
        v0.y = p2.y - p0.y;
        v1.x = p1.x - p0.x;
        v1.y = p1.y - p0.y;

        // dot products np.dot(v0.T, v0)
        dot00 = v0.x * v0.x + v0.y * v0.y;
        dot01 = v0.x * v1.x + v0.y * v1.y;
        dot11 = v1.x * v1.x + v1.y * v1.y;

        // barycentric coordinates
        barycentric = dot00 * dot11 - dot01 * dot01;
        inverDeno = barycentric == 0 ? 0 : 1 / barycentric;

        for (y = y_min; y <= y_max; ++y)
        {
            for (x = x_min; x <= x_max; ++x)
            {
                v2.x = x - p0.x;
                v2.y = y - p0.y;

                dot02 = v0.x * v2.x + v0.y * v2.y;
                dot12 = v1.x * v2.x + v1.y * v2.y;

                u = (dot11 * dot02 - dot01 * dot12) * inverDeno;
                v = (dot00 * dot12 - dot01 * dot02) * inverDeno;
                t = 1 - u - v;

                // judge is_point_in_tri by below line of code
                if (u >= 0 && v >= 0 && t > 0)
                {
                    p_depth = t * p0_depth + v * p1_depth + u * p2_depth;
                    color_index = y * w + x;

                    if (p_depth > depth_buffer[color_index])
                    {
                        for (k = 0; k < c; ++k)
                        {
                            p_color = t * colors[tri_p0_ind + k];
                            p_color += v * colors[tri_p1_ind + k];
                            p_color += u * colors[tri_p2_ind + k];
                            p_color *= 255;

                            image[color_index * c + k] = (unsigned char)p_color;
                        }

                        depth_buffer[color_index] = p_depth;
                    }
                }
            }
        }
    }

    free(depth_buffer);
}

/*
 * Directly get normal of vertices, which can be regraded as a combination of _get_tri_normal and _get_ver_normal
 */
void _render(const float *vertices, int nver,
             const int *triangles, int ntri,
             const float *light,
             const float *directional,
             const float *ambient,
             unsigned char *image, int h, int w, int c)
{
    int tri_p0_ind, tri_p1_ind, tri_p2_ind;
    float v1x, v1y, v1z, v2x, v2y, v2z;
    float resx, resy, resz;

    float *ver_normal = (float *)calloc(3 * nver, sizeof(float));
    float *colors = (float *)malloc(3 * nver * sizeof(float));

    for (int i = 0; i < ntri; i++)
    {
        tri_p0_ind = triangles[3 * i];
        tri_p1_ind = triangles[3 * i + 1];
        tri_p2_ind = triangles[3 * i + 2];

        // counter clockwise order
        v1x = vertices[tri_p1_ind] - vertices[tri_p0_ind];
        v1y = vertices[tri_p1_ind + 1] - vertices[tri_p0_ind + 1];
        v1z = vertices[tri_p1_ind + 2] - vertices[tri_p0_ind + 2];

        v2x = vertices[tri_p2_ind] - vertices[tri_p0_ind];
        v2y = vertices[tri_p2_ind + 1] - vertices[tri_p0_ind + 1];
        v2z = vertices[tri_p2_ind + 2] - vertices[tri_p0_ind + 2];

        resx = v1y * v2z - v1z * v2y;
        resy = v1z * v2x - v1x * v2z;
        resz = v1x * v2y - v1y * v2x;

        ver_normal[tri_p0_ind] += resx;
        ver_normal[tri_p1_ind] += resx;
        ver_normal[tri_p2_ind] += resx;

        ver_normal[tri_p0_ind + 1] += resy;
        ver_normal[tri_p1_ind + 1] += resy;
        ver_normal[tri_p2_ind + 1] += resy;

        ver_normal[tri_p0_ind + 2] += resz;
        ver_normal[tri_p1_ind + 2] += resz;
        ver_normal[tri_p2_ind + 2] += resz;
    }

    // normalizing
    float nx, ny, nz, det;
    float max_x = -1.0e8, max_y = -1.0e8, max_z = -1.0e8;
    float min_x = 1.0e8, min_y = 1.0e8, min_z = 1.0e8;
    float mean_x = 0.0, mean_y = 0.0, mean_z = 0.0;
    for (int i = 0; i < nver; ++i)
    {
        nx = ver_normal[3 * i];
        ny = ver_normal[3 * i + 1];
        nz = ver_normal[3 * i + 2];

        det = sqrt(nx * nx + ny * ny + nz * nz);
        if (det <= 0)
            det = 1e-6;

        ver_normal[3 * i] /= det;
        ver_normal[3 * i + 1] /= det;
        ver_normal[3 * i + 2] /= det;

        mean_x += nx;
        mean_y += ny;
        mean_z += nz;

        max_x = max(max_x, nx);
        max_y = max(max_y, ny);
        max_z = max(max_z, nz);

        min_x = min(min_x, nx);
        min_y = min(min_y, ny);
        min_z = min(min_z, nz);
    }

    mean_x /= nver;
    mean_y /= nver;
    mean_z /= nver;

    float cos_sum;

    for (int i = 0; i < nver; ++i)
    {
        colors[3 * i] = vertices[3 * i];
        colors[3 * i + 1] = vertices[3 * i + 1];
        colors[3 * i + 2] = vertices[3 * i + 2];

        colors[3 * i] -= mean_x;
        colors[3 * i] /= max_x - min_x;

        colors[3 * i + 1] -= mean_y;
        colors[3 * i + 1] /= max_y - min_y;

        colors[3 * i + 2] -= mean_z;
        colors[3 * i + 2] /= max_z - min_z;

        nx = light[0] - colors[3 * i];
        ny = light[1] - colors[3 * i + 1];
        nz = light[2] - colors[3 * i + 2];

        det = sqrt(nx * nx + ny * ny + nz * nz);
        if (det <= 0)
            det = 1e-6;

        colors[3 * i] = nx / det;
        colors[3 * i + 1] = ny / det;
        colors[3 * i + 2] = nz / det;

        colors[3 * i] *= ver_normal[3 * i];
        colors[3 * i + 1] *= ver_normal[3 * i + 1];
        colors[3 * i + 2] *= ver_normal[3 * i + 2];

        cos_sum = colors[3 * i] + colors[3 * i + 1] + colors[3 * i + 2];

        colors[3 * i] = clip(cos_sum * directional[0] + ambient[0], 0, 1);
        colors[3 * i + 1] = clip(cos_sum * directional[1] + ambient[1], 0, 1);
        colors[3 * i + 2] = clip(cos_sum * directional[2] + ambient[2], 0, 1);
    }

    _rasterize(image, vertices, triangles, colors, ntri, h, w, c);

    free(ver_normal);
    free(colors);
}
