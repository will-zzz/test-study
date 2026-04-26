/*
 *  Copyright 2026 Will Zakielarz
 */

#include "wmz_canvas.h"
#include "include/GPath.h"
#include "include/GPixel.h"
#include "include/GMath.h"
#include "include/GRect.h"
#include "include/GColor.h"
#include "include/GPaint.h"
#include "include/GShader.h"
#include "include/GBitmap.h"
#include "include/GPathBuilder.h"
#include <vector>
#include <algorithm>

// Helper: Distance from point to line segment (scary math ahead)
static float distance_to_segment(GPoint p, GPoint a, GPoint b) {
    // Slope
    float dx = b.x - a.x;
    float dy = b.y - a.y;
    
    // Normalize
    float den = dx * dx + dy * dy;
    
    // If point not line, return distance to point
    if (den == 0) {
        float px = p.x - a.x;
        float py = p.y - a.y;
        return std::sqrt(px * px + py * py);
    }
    
    // Project point onto line segment because scary math says so
    float apx = p.x - a.x;
    float apy = p.y - a.y;
    float t = (apx * dx + apy * dy) / den;
    
    // Clamp t
    if (t < 0) t = 0;
    if (t > 1) t = 1;
    
    // Find closest point on segment
    GPoint closest = {a.x + t * dx, a.y + t * dy};
    
    // Return distance from p
    float pdx = p.x - closest.x;
    float pdy = p.y - closest.y;

    return std::sqrt(pdx * pdx + pdy * pdy);
}

// Helper: GColor (unpremul) -> GPixel (premul)
static GPixel color_to_pixel(const GColor& color) {
    // Clamp color components (0 - 1)
    float a = std::max(0.0f, std::min(1.0f, color.a));
    float r = std::max(0.0f, std::min(1.0f, color.r));
    float g = std::max(0.0f, std::min(1.0f, color.g));
    float b = std::max(0.0f, std::min(1.0f, color.b));
    
    // Turn alpha to 0 - 255
    int alpha = GRoundToInt(a * 255);
    // Premultiply by alpha
    int red = GRoundToInt(r * a * 255);
    int green = GRoundToInt(g * a * 255);
    int blue = GRoundToInt(b * a * 255);
    
    // Return GPixel
    return GPixel_PackARGB(alpha, red, green, blue);
}

// Shared blend helper (used by both canvas and shader)
#include "wmz_blend.h"

// Helper: apply alpha to premul GPixel
static GPixel apply_alpha(GPixel src, float alpha) {
    if (alpha >= 1.0f) {
        return src;
    }
    int a = GRoundToInt(GPixel_GetA(src) * alpha);
    int r = GRoundToInt(GPixel_GetR(src) * alpha);
    int g = GRoundToInt(GPixel_GetG(src) * alpha);
    int b = GRoundToInt(GPixel_GetB(src) * alpha);
    return GPixel_PackARGB(a, r, g, b);
}

// Lerps 3 vertex colors across triangle
class ColorLerp : public GShader {
public:
    ColorLerp(GPoint p0, GPoint p1, GPoint p2, GColor c0, GColor c1, GColor c2)
    : fP0(p0), fP1(p1), fP2(p2), fC0(c0), fC1(c1), fC2(c2) {}

    bool isOpaque() override {
        return fC0.a >= 1.0f && fC1.a >= 1.0f && fC2.a >= 1.0f;
    }

    std::unique_ptr<Context> makeContext(const GMatrix& ctm) override {
        // Map triangle -> local triangle verts
        GMatrix P({fP1.x - fP0.x, fP1.y - fP0.y},
                  {fP2.x - fP0.x, fP2.y - fP0.y},
                  {fP0.x, fP0.y});
        // Combine with CTM and invert
        auto inv = GMatrix::Concat(ctm, P).invert();

        if (!inv) return nullptr;

        return std::make_unique<ColorLerpCtx>(fC0, fC1, fC2, *inv);
    }

private:
    GPoint fP0, fP1, fP2;
    GColor fC0, fC1, fC2;

    class ColorLerpCtx : public Context {
    public:
        ColorLerpCtx(GColor c0, GColor c1, GColor c2, const GMatrix& inv)
        : fC0(c0), fC1(c1), fC2(c2), fInverse(inv) {}

        void shadeRow(int x, int y, int count, GPixel row[]) override {
            // Map first pixel center to barycentric
            GPoint loc = fInverse * GPoint{x + 0.5f, y + 0.5f};
            GVector step = fInverse.e0();

            for (int i = 0; i < count; i++) {
                // u = weight for C1, v = weight for C2, w = weight for C0
                float u = loc.x;
                float v = loc.y;
                float w = 1.0f - u - v;

                // Lerp colors (unpremul)
                GColor c;
                c.r = w * fC0.r + u * fC1.r + v * fC2.r;
                c.g = w * fC0.g + u * fC1.g + v * fC2.g;
                c.b = w * fC0.b + u * fC1.b + v * fC2.b;
                c.a = w * fC0.a + u * fC1.a + v * fC2.a;

                row[i] = color_to_pixel(c);
                loc.x += step.x;
                loc.y += step.y;
            }
        }

    private:
        GColor fC0, fC1, fC2;
        GMatrix fInverse;
    };
};

// Remaps another shader's coords for texture mapping
class TexRemap : public GShader {
public:
    TexRemap(GShader* original, const GMatrix& extraLocal)
    : fOriginal(original), fExtraLocal(extraLocal) {}

    bool isOpaque() override {
        return fOriginal->isOpaque();
    }

    std::unique_ptr<Context> makeContext(const GMatrix& ctm) override {
        // Add extra matrix to start so shader maps through texture coords
        return fOriginal->makeContext(GMatrix::Concat(ctm, fExtraLocal));
    }

private:
    GShader* fOriginal;
    GMatrix fExtraLocal;
};

// Mult outputs of two shaders
class ShaderMultiply : public GShader {
public:
    ShaderMultiply(std::shared_ptr<GShader> s0, std::shared_ptr<GShader> s1)
    : fShader0(std::move(s0)), fShader1(std::move(s1)) {}

    bool isOpaque() override {
        return fShader0->isOpaque() && fShader1->isOpaque();
    }

    std::unique_ptr<Context> makeContext(const GMatrix& ctm) override {
        auto ctx0 = fShader0->makeContext(ctm);
        auto ctx1 = fShader1->makeContext(ctm);

        if (!ctx0 || !ctx1) {
            return nullptr;
        }
        return std::make_unique<ShaderMultiplyCtx>(std::move(ctx0), std::move(ctx1));
    }

private:
    std::shared_ptr<GShader> fShader0, fShader1;

    class ShaderMultiplyCtx : public Context {
    public:
        ShaderMultiplyCtx(std::unique_ptr<Context> c0, std::unique_ptr<Context> c1)
            : fCtx0(std::move(c0)), fCtx1(std::move(c1)) {}

        void shadeRow(int x, int y, int count, GPixel row[]) override {
            // Get pixels from both shaders
            std::vector<GPixel> tmp(count);
            fCtx0->shadeRow(x, y, count, row);
            fCtx1->shadeRow(x, y, count, tmp.data());

            // Mult premul components (a * b) / 255
            for (int i = 0; i < count; i++) {
                int a = GRoundToInt(GPixel_GetA(row[i]) * GPixel_GetA(tmp[i]) / 255.0f);
                int r = GRoundToInt(GPixel_GetR(row[i]) * GPixel_GetR(tmp[i]) / 255.0f);
                int g = GRoundToInt(GPixel_GetG(row[i]) * GPixel_GetG(tmp[i]) / 255.0f);
                int b = GRoundToInt(GPixel_GetB(row[i]) * GPixel_GetB(tmp[i]) / 255.0f);

                row[i] = GPixel_PackARGB(a, r, g, b);
            }
        }

    private:
        std::unique_ptr<Context> fCtx0, fCtx1;
    };
};

// Helper: blend and place pixel (premul)
void WMZCanvas::place_pixel(int x, int y, GPixel src, GBlendMode blend_mode) {
    GPixel dest = *fDevice.getAddr(x, y);
    GPixel result = apply_blend(src, dest, blend_mode);
    *fDevice.getAddr(x, y) = result;
}

// Turn whole image to color
void WMZCanvas::clear(const GColor& color) {
    // Color -> premul pixel
    GPixel pixel = color_to_pixel(color);

    // y then x for standard of rendering across screen and going down
    for (int y = 0; y < fDevice.height(); y++) {
        for (int x = 0; x < fDevice.width(); x++) {
            place_pixel(x, y, pixel, GBlendMode::kSrc);
        }
    }
}

void WMZCanvas::drawLine(GPoint a, GPoint b, const GPaint& paint) {
    float width = paint.lineWidth();
    
    // Hairline
    if (width < 0) {
        // Rasterize in device coords
        fCTM.mapPoints(&a, 1);
        fCTM.mapPoints(&b, 1);

        // Make shader context
        GShader* shader = paint.peekShader();
        std::unique_ptr<GShader::Context> shaderCtx;
        if (shader) {
            shaderCtx = shader->makeContext(fCTM);
            if (!shaderCtx) {
                return;
            };
        }

        // If there's no shader we use this
        GPixel solidPixel = color_to_pixel(paint.color());

        // Get bounding box (we only check in this instead of whole bitmap)
        int left = std::max(0, GFloorToInt(std::min(a.x, b.x)));
        int right = std::min(fDevice.width() - 1, GFloorToInt(std::max(a.x, b.x)));
        int top = std::max(0, GFloorToInt(std::min(a.y, b.y)));
        int bottom = std::min(fDevice.height() - 1, GFloorToInt(std::max(a.y, b.y)));
        
        for (int y = top; y <= bottom; y++) {
            for (int x = left; x <= right; x++) {
                // Pixel center
                GPoint pc = {x + 0.5f, y + 0.5f};
                
                float distance = distance_to_segment(pc, a, b);
                
                // Circle method
                float radius = 0.5f;
                if (distance <= radius) {
                    // Shader
                    if (shaderCtx) {
                        GPixel scratch[1];
                        shaderCtx->shadeRow(x, y, 1, scratch);
                        place_pixel(x, y, apply_alpha(scratch[0], paint.alpha()), paint.blendMode());
                    // Plain color
                    } else {
                        place_pixel(x, y, solidPixel, paint.blendMode());
                    }
                }
            }
        }
    }
    // Not hairline
    else {
        // Transform to device space
        fCTM.mapPoints(&a, 1);
        fCTM.mapPoints(&b, 1);

        // Make shader context
        GShader* shader = paint.peekShader();
        std::unique_ptr<GShader::Context> shaderCtx;
        if (shader) {
            shaderCtx = shader->makeContext(fCTM);
            if (!shaderCtx) {
                return;
            };
        }

        // If no shader we use this
        GPixel solidPixel = color_to_pixel(paint.color());

        const float half_w = width * 0.5f;
        GPoint ab = {b.x - a.x, b.y - a.y};
        float len = std::sqrt(ab.x * ab.x + ab.y * ab.y);

        // Treat round as a disk, others as nothing
        if (len == 0) {
            if (paint.capType() != GCapType::kRound) {
                return;
            }

            // Bounding box of endpoint disk
            int left = std::max(0, GFloorToInt(a.x - half_w));
            int right = std::min(fDevice.width() - 1, GFloorToInt(a.x + half_w));
            int top = std::max(0, GFloorToInt(a.y - half_w));
            int bottom = std::min(fDevice.height() - 1, GFloorToInt(a.y + half_w));

            for (int y = top; y <= bottom; y++) {
                for (int x = left; x <= right; x++) {
                    // Pixel center
                    GPoint pc = {x + 0.5f, y + 0.5f};
                    float dx = pc.x - a.x;
                    float dy = pc.y - a.y;
                    
                    // Inside round cap if center within radius
                    if (dx * dx + dy * dy <= half_w * half_w) {
                        if (shaderCtx) {
                            GPixel pix[1];
                            shaderCtx->shadeRow(x, y, 1, pix);
                            place_pixel(x, y, apply_alpha(pix[0], paint.alpha()), paint.blendMode());
                        } else {
                            place_pixel(x, y, solidPixel, paint.blendMode());
                        }
                    }
                }
            }
            return;
        }

        // Extends endpoints by half width along tangent
        float tx = ab.x / len;
        float ty = ab.y / len;
        float extend = paint.capType() == GCapType::kSquare ? half_w : 0.0f;

        // Segment endpoints after cap extension
        GPoint s0 = {a.x - tx * extend, a.y - ty * extend};
        GPoint s1 = {b.x + tx * extend, b.y + ty * extend};
        // Segment direction vector
        GPoint sv = {s1.x - s0.x, s1.y - s0.y};

        float slen2 = sv.x * sv.x + sv.y * sv.y;
        float inv = 1.0f / std::sqrt(slen2);

        // Bounding box
        float minx = std::min(s0.x, s1.x) - half_w;
        float maxx = std::max(s0.x, s1.x) + half_w;
        float miny = std::min(s0.y, s1.y) - half_w;
        float maxy = std::max(s0.y, s1.y) + half_w;

        int left = std::max(0, GFloorToInt(minx));
        int right = std::min(fDevice.width() - 1, GFloorToInt(maxx));
        int top = std::max(0, GFloorToInt(miny));
        int bottom = std::min(fDevice.height() - 1, GFloorToInt(maxy));

        for (int y = top; y <= bottom; y++) {
            for (int x = left; x <= right; x++) {
                GPoint pc = {x + 0.5f, y + 0.5f};

                bool inside = false;
                if (paint.capType() == GCapType::kRound) {
                    inside = distance_to_segment(pc, a, b) <= half_w;
                } else {
                    // Square
                    GPoint w = {pc.x - s0.x, pc.y - s0.y};
                    float t = (w.x * sv.x + w.y * sv.y) / slen2;
                    if (t >= 0.0f && t <= 1.0f) {
                        float cross = sv.x * w.y - sv.y * w.x;
                        float perp = std::fabs(cross) * inv;
                        inside = perp <= half_w;
                    }
                }

                if (!inside) {
                    continue;
                }

                if (shaderCtx) {
                    GPixel pix[1];
                    shaderCtx->shadeRow(x, y, 1, pix);
                    place_pixel(x, y, apply_alpha(pix[0], paint.alpha()), paint.blendMode());
                } else {
                    place_pixel(x, y, solidPixel, paint.blendMode());
                }
            }
        }
    }
}

// Edge for scan conversion
struct Edge {
    float top;
    float bottom;
    float currX;
    float slope;
    int winding;
};

// Helper: Linear interpolation
static GPoint lerp_point(GPoint a, GPoint b, float t) {
    return {a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t};
}

// Helper: find Bezier at parameter t (quadratic)
static GPoint quad_eval(GPoint p0, GPoint p1, GPoint p2, float t) {
    float u = 1.0f - t;
    float a = u * u;
    float b = 2.0f * u * t;
    float c = t * t;
    return {
        a * p0.x + b * p1.x + c * p2.x,
        a * p0.y + b * p1.y + c * p2.y,
    };
}

// Helper: find Bezier at parameter t (cubic)
static GPoint cubic_eval(GPoint p0, GPoint p1, GPoint p2, GPoint p3, float t) {
    float u = 1.0f - t;
    float a = u * u * u;
    float b = 3.0f * u * u * t;
    float c = 3.0f * u * t * t;
    float d = t * t * t;
    return {
        a * p0.x + b * p1.x + c * p2.x + d * p3.x,
        a * p0.y + b * p1.y + c * p2.y + d * p3.y,
    };
}

// Helper: Control point is close enough to end-point (flat)
static bool quad_is_flat(GPoint p0, GPoint p1, GPoint p2, float tol) {
    return distance_to_segment(p1, p0, p2) <= tol;
}

// Helper: Both cubic controls are close enough to end-point (cubic)
static bool cubic_is_flat(GPoint p0, GPoint p1, GPoint p2, GPoint p3, float tol) {
    return std::max(distance_to_segment(p1, p0, p3), distance_to_segment(p2, p0, p3)) <= tol;
}

// Helper: segment count
static int quad_segment_count(GPoint p0, GPoint p1, GPoint p2, float tol) {
    struct QuadSeg { GPoint p0, p1, p2; };
    int kMaxSegs = 1024;

    // Stack (avoids recursion)
    std::vector<QuadSeg> stack;
    stack.push_back({p0, p1, p2});
    int segs = 0;

    while (!stack.empty() && segs < kMaxSegs) {
        QuadSeg q = stack.back();
        stack.pop_back();

        if (quad_is_flat(q.p0, q.p1, q.p2, tol)) {
            // Flat contributes one line piece
            segs++;
            continue;
        }

        // Maths things
        GPoint p01 = lerp_point(q.p0, q.p1, 0.5f);
        GPoint p12 = lerp_point(q.p1, q.p2, 0.5f);
        GPoint p012 = lerp_point(p01, p12, 0.5f);

        stack.push_back({p012, p12, q.p2});
        stack.push_back({q.p0, p01, p012});
    }

    return std::max(1, segs);
}

static int cubic_segment_count(GPoint p0, GPoint p1, GPoint p2, GPoint p3, float tol) {
    struct CubicSeg { GPoint p0, p1, p2, p3; };
    constexpr int kMaxSegs = 1024;

    // Stack (avoids recursion)
    std::vector<CubicSeg> stack;
    stack.push_back({p0, p1, p2, p3});
    int segs = 0;

    while (!stack.empty() && segs < kMaxSegs) {
        CubicSeg c = stack.back();
        stack.pop_back();

        if (cubic_is_flat(c.p0, c.p1, c.p2, c.p3, tol)) {
            // Flat contributes one line piece
            segs++;
            continue;
        }

        // Split cubic in half
        GPoint p01 = lerp_point(c.p0, c.p1, 0.5f);
        GPoint p12 = lerp_point(c.p1, c.p2, 0.5f);
        GPoint p23 = lerp_point(c.p2, c.p3, 0.5f);
        GPoint p012 = lerp_point(p01, p12, 0.5f);
        GPoint p123 = lerp_point(p12, p23, 0.5f);
        GPoint p0123 = lerp_point(p012, p123, 0.5f);

        stack.push_back({p0123, p123, p23, c.p3});
        stack.push_back({c.p0, p01, p012, p0123});
    }

    return std::max(1, segs);
}

// Helper: build edge from two points, clip to canvas height
static bool make_edge(GPoint p0, GPoint p1, int canvasH, Edge& edge) {
    // Skip horizontal
    if (p0.y == p1.y) {
        return false;
    }

    // Winding: 1 if down, -1 if going up
    int winding = 1;
    if (p0.y > p1.y) {
        std::swap(p0, p1);
        winding = -1;
    }

    // Off screen
    if (p1.y <= 0 || p0.y >= canvasH) {
        return false;
    }

    float dy = p1.y - p0.y;
    float slope = (p1.x - p0.x) / dy;

    // Clip top edge
    if (p0.y < 0) {
        p0.x += slope * (0 - p0.y);
        p0.y = 0;
    }
    // Clip bottom edge
    if (p1.y > canvasH) {
        p1.y = canvasH;
    }

    edge.top = p0.y;
    edge.bottom = p1.y;
    edge.currX = p0.x;
    edge.slope = slope;
    edge.winding = winding;
    return true;
}

void WMZCanvas::drawPath(const GPath& path, const GPaint& paint) {
    // Transform path by CTM
    auto transformed = path.transform(fCTM);

    std::vector<Edge> edges;
    GPoint pts[GPath::kMaxNextPoints];
    GPath::Edger edger(*transformed);
    float tol = std::max(0.0001f, paint.curveTolerance());
    // Walk the path with Edger
    while (auto v = edger.next(pts)) {
        if (v.value() == GPathVerb::kLine) {
            Edge e;
            if (make_edge(pts[0], pts[1], fDevice.height(), e)) {
                edges.push_back(e);
            }
        } else if (v.value() == GPathVerb::kQuad) {
            int segs = quad_segment_count(pts[0], pts[1], pts[2], tol);
            GPoint prev = pts[0];
            for (int i = 1; i <= segs; i++) {
                float t = (float)i / segs;
                GPoint curr = quad_eval(pts[0], pts[1], pts[2], t);
                Edge e;
                if (make_edge(prev, curr, fDevice.height(), e)) {
                    edges.push_back(e);
                }
                prev = curr;
            }
        } else if (v.value() == GPathVerb::kCubic) {
            int segs = cubic_segment_count(pts[0], pts[1], pts[2], pts[3], tol);
            GPoint prev = pts[0];
            for (int i = 1; i <= segs; i++) {
                float t = (float)i / segs;
                GPoint curr = cubic_eval(pts[0], pts[1], pts[2], pts[3], t);
                Edge e;
                if (make_edge(prev, curr, fDevice.height(), e)) {
                    edges.push_back(e);
                }
                prev = curr;
            }
        }
    }

    if (edges.empty()) {
        return;
    }

    // Make shader context
    GShader* shader = paint.peekShader();
    std::unique_ptr<GShader::Context> shaderCtx;
    if (shader) {
        shaderCtx = shader->makeContext(fCTM);
        if (!shaderCtx) {
            return;
        }
    }

    // If no shader we use this
    GPixel solidPixel = color_to_pixel(paint.color());

    // Find y range across all edges
    float y_min = edges[0].top, y_max = edges[0].bottom;
    for (size_t i = 1; i < edges.size(); i++) {
        y_min = std::min(y_min, edges[i].top);
        y_max = std::max(y_max, edges[i].bottom);
    }
    int y_start = std::max(0, GFloorToInt(y_min));
    int y_end = std::min(fDevice.height(), GCeilToInt(y_max));

    // Scan lines
    for (int y = y_start; y < y_end; y++) {
        float yc = y + 0.5f;

        // Where edge hits this scanline with its winding
        struct Crossing {
            float x;
            int winding;
        };
        std::vector<Crossing> crossings;

        for (const auto& e : edges) {
            // Edge active on this scanline?
            if (yc < e.top || yc >= e.bottom) continue;
            // Where does it cross
            float x = e.currX + (yc - e.top) * e.slope;
            crossings.push_back({x, e.winding});
        }

        // Sort left to right
        std::sort(crossings.begin(), crossings.end(),
            [](const Crossing& a, const Crossing& b) { return a.x < b.x; });

        // Walk crossings with winding
        int winding = 0;
        size_t span_start = 0;
        for (size_t i = 0; i < crossings.size(); i++) {
            int old_winding = winding;
            winding += crossings[i].winding;

            // 0 -> non-zero: in filled area
            if (old_winding == 0 && winding != 0) {
                span_start = i;
            }
            // non-zero -> 0: out of filled area, draw the span
            if (old_winding != 0 && winding == 0) {
                int x_left = std::max(0, GCeilToInt(crossings[span_start].x - 0.5f));
                int x_right = std::min(fDevice.width() - 1, GFloorToInt(crossings[i].x - 0.5f));
                int spanCount = x_right - x_left + 1;
                if (spanCount <= 0) {
                    continue;
                }
                if (shaderCtx) {
                    std::vector<GPixel> scratch(spanCount);
                    shaderCtx->shadeRow(x_left, y, spanCount, scratch.data());
                    for (int j = 0; j < spanCount; j++) {
                        GPixel src = apply_alpha(scratch[j], paint.alpha());
                        place_pixel(x_left + j, y, src, paint.blendMode());
                    }
                } else {
                    for (int x = x_left; x <= x_right; x++)
                        place_pixel(x, y, solidPixel, paint.blendMode());
                }
            }
        }
    }
}

void WMZCanvas::drawRect(const GRect& rect, const GPaint& paint) {
    GPoint pts[4] = { rect.TL(), rect.TR(), rect.BR(), rect.BL() };
    drawConvexPolygon(pts, 4, paint);
}

// Helper: clip edge (From yt video, notes)
static void clip_edge(std::vector<GPoint>& vertices, float edge_val, bool vertical, bool keep_above) {
    std::vector<GPoint> out;
    int n = (int)vertices.size();

    for (int i = 0; i < n; i++) {
        // Edge from vertex i to next (wrap with % n)
        GPoint P0 = vertices[i];
        GPoint P1 = vertices[(i + 1) % n];

        // Is each endpoint on "keep" side?
        bool in0, in1;
        if (vertical) {
            in0 = keep_above ? (P0.x >= edge_val) : (P0.x < edge_val);
            in1 = keep_above ? (P1.x >= edge_val) : (P1.x < edge_val);
        } else {
            in0 = keep_above ? (P0.y >= edge_val) : (P0.y < edge_val);
            in1 = keep_above ? (P1.y >= edge_val) : (P1.y < edge_val);
        }

        if (in0 && in1) {
            out.push_back(P1);
            continue;
        }
        if (!in0 && !in1) {
            continue;
        }

        // find where edge crosses clip line
        float den = vertical ? (P1.x - P0.x) : (P1.y - P0.y);
        if (den == 0) {
            if (in1) {
                out.push_back(P1);
            }
            continue;
        }
        float t = vertical ? (edge_val - P0.x) / den : (edge_val - P0.y) / den;
        if (t < 0 || t > 1) {
            if (in1) {
                out.push_back(P1);
            }
            continue;
        }

        // Crossing point on the segment
        GPoint cross;
        if (vertical) {
            cross.x = edge_val;
            cross.y = P0.y + t * (P1.y - P0.y);
        } else {
            cross.x = P0.x + t * (P1.x - P0.x);
            cross.y = edge_val;
        }

        if (in0 && !in1)
            out.push_back(cross);
        else {
            out.push_back(cross);
            out.push_back(P1);
        }
    }
    vertices = out;
}

// Helper: clip polygon to rect
static int clip_polygon_to_rect(std::vector<GPoint>& vertices, float clipL, float clipT, float clipR, float clipB) {
    clip_edge(vertices, clipT, false, true);
    if (vertices.empty()) return 0;
    clip_edge(vertices, clipB, false, false);
    if (vertices.empty()) return 0;
    clip_edge(vertices, clipL, true, true);
    if (vertices.empty()) return 0;
    clip_edge(vertices, clipR, true, false);
    if (vertices.empty()) return 0;

    return vertices.size();
}

void WMZCanvas::drawConvexPolygon(const GPoint points[], int count, const GPaint& paint) {
    if (count < 3) return;
    // Had to look up vector for this
    std::vector<GPoint> vertices(count);
    // Transform
    fCTM.mapPoints(vertices.data(), points, count);
    int n = clip_polygon_to_rect(vertices, 0, 0, (float)fDevice.width(), (float)fDevice.height());
    if (n < 3) return;

    GShader* shader = paint.peekShader();
    std::unique_ptr<GShader::Context> shaderCtx;
    if (shader) {
        shaderCtx = shader->makeContext(fCTM);
        if (!shaderCtx) {
            return;
        }
    }

    GPixel solidPixel = color_to_pixel(paint.color());

    // Start and end rows
    float y_min = vertices[0].y, y_max = vertices[0].y;
    for (int i = 1; i < n; i++) {
        y_min = std::min(y_min, vertices[i].y);
        y_max = std::max(y_max, vertices[i].y);
    }
    int y_start = std::max(0, GCeilToInt(y_min - 0.5f));
    int y_end = std::min(fDevice.height() - 1, GFloorToInt(y_max - 0.5f));
    
    // Each line, find where horizontal line hits the polygon edges
    // We get 0 or 2 crossings (if clip works right) and then fill between them
    for (int y = y_start; y <= y_end; y++) {
        float yc = y + 0.5f;
        float xs[2];  // x-coords where hits polygon (left, right)
        int num_xs = 0;
        for (int i = 0; i < n; i++) {
            GPoint P0 = vertices[i];
            GPoint P1 = vertices[(i + 1) % n]; // % n so last edge closes to vertex 0
            // Get slope
            float dy = P1.y - P0.y;
            // Skip horizontal edge
            if (dy == 0) continue;

            float t = (yc - P0.y) / dy;  // where edge crosses yc
            // If on edge
            if (t >= 0 && t <= 1) {
                // Lerp..?
                float x_cross = P0.x + t * (P1.x - P0.x);
                if (num_xs == 0) {
                    xs[0] = x_cross;
                    num_xs = 1;
                } else if (num_xs == 1) {
                    xs[1] = x_cross;
                    // Keep left crossing in xs[0], right in xs[1] (swap)
                    if (xs[1] < xs[0]) { float tmp = xs[0]; xs[0] = xs[1]; xs[1] = tmp; }
                    num_xs = 2;
                }
            }
        }
        if (num_xs < 2) continue;
        // Clamp and draw row
        int x_left = std::max(0, GCeilToInt(xs[0] - 0.5f));
        int x_right = std::min(fDevice.width() - 1, GFloorToInt(xs[1] - 0.5f));
        int spanCount = x_right - x_left + 1;

        if (shaderCtx) {
            std::vector<GPixel> scratch(spanCount);
            shaderCtx->shadeRow(x_left, y, spanCount, scratch.data());
            for (int i = 0; i < spanCount; i++) {
                GPixel src = apply_alpha(scratch[i], paint.alpha());
                place_pixel(x_left + i, y, src, paint.blendMode());
            }
        } else {
            for (int x = x_left; x <= x_right; x++)
                place_pixel(x, y, solidPixel, paint.blendMode());
        }
    }
}

void WMZCanvas::drawMesh(const GPoint verts[], const GColor colors[], const GPoint texs[], int count, const int indices[], const GPaint& paint) {
    int n = 0;
    for (int i = 0; i < count; i++) {
        // Triangle vertex indices
        int i0 = indices[n + 0];
        int i1 = indices[n + 1];
        int i2 = indices[n + 2];
        n += 3;

        // Triangle vertices (local space)
        GPoint triPts[3] = {verts[i0], verts[i1], verts[i2]};

        GPaint triPaint(paint);

        // Colors only -> lerp vertex colors
        if (colors && !texs) {
            auto colorShader = std::make_shared<ColorLerp>(
                triPts[0], triPts[1], triPts[2],
                colors[i0], colors[i1], colors[i2]);
            triPaint.setShader(colorShader);
        }
        // Texs only -> remap paint's shader thru texture coordinates
        else if (texs && !colors) {
            GPoint t0 = texs[i0], t1 = texs[i1], t2 = texs[i2];
            // Unit triangle -> local triangle verts
            GMatrix P({triPts[1].x - triPts[0].x, triPts[1].y - triPts[0].y},
                      {triPts[2].x - triPts[0].x, triPts[2].y - triPts[0].y},
                      {triPts[0].x, triPts[0].y});

            // Unit triangle -> tex coords
            GMatrix T({t1.x - t0.x, t1.y - t0.y},
                      {t2.x - t0.x, t2.y - t0.y},
                      {t0.x, t0.y});
            auto invT = T.invert();

            if (!invT) continue;

            // Maps local verts -> unit -> tex
            GMatrix extraLocal = GMatrix::Concat(P, *invT);
            auto proxy = std::make_shared<TexRemap>(paint.peekShader(), extraLocal);

            triPaint.setShader(proxy);
        }
        // Both colors AND texs -> mult shader outputs
        else if (colors && texs) {
            // Lerp vertex colors
            auto colorShader = std::make_shared<ColorLerp>(
                triPts[0], triPts[1], triPts[2],
                colors[i0], colors[i1], colors[i2]);

            // Remap paint's shader thru texture coordinates
            GPoint t0 = texs[i0], t1 = texs[i1], t2 = texs[i2];
            GMatrix P({triPts[1].x - triPts[0].x, triPts[1].y - triPts[0].y},
                      {triPts[2].x - triPts[0].x, triPts[2].y - triPts[0].y},
                      {triPts[0].x, triPts[0].y});
            GMatrix T({t1.x - t0.x, t1.y - t0.y},
                      {t2.x - t0.x, t2.y - t0.y},
                      {t0.x, t0.y});
            auto invT = T.invert();

            if (!invT) continue;

            // Maps local verts -> unit -> tex
            GMatrix extraLocal = GMatrix::Concat(P, *invT);
            auto texShader = std::make_shared<TexRemap>(paint.peekShader(), extraLocal);

            // Multiply tex shader * color shader
            auto multiply = std::make_shared<ShaderMultiply>(texShader, colorShader);
            triPaint.setShader(multiply);
        }
        // Neither -> just draw with paint as-is
        drawConvexPolygon(triPts, 3, triPaint);
    }
}

void WMZCanvas::drawQuad(const GPoint verts[4], const GColor colors[4], const GPoint texs[4], int level, const GPaint& paint) {
    // n = subdivisions per side
    int n = level + 1;
    int vertCount = (n + 1) * (n + 1);

    // [0]=top left, [1]=top right, [2]=bottom right, [3]=bottom left
    std::vector<GPoint> meshVerts(vertCount);
    std::vector<GColor> meshColors;
    std::vector<GPoint> meshTexs;
    // colors + texs -> mult shader outputs
    if (colors) meshColors.resize(vertCount);
    if (texs) meshTexs.resize(vertCount);

    for (int row = 0; row <= n; row++) {
        float v = (float)row / n;
        for (int col = 0; col <= n; col++) {
            float u = (float)col / n;
            int idx = row * (n + 1) + col;

            // Weights for four corners
            float w00 = (1 - u) * (1 - v);  // top left
            float w10 = u * (1 - v);         // top right
            float w11 = u * v;               // bottom right
            float w01 = (1 - u) * v;         // bottom left

            meshVerts[idx] = {
                w00 * verts[0].x + w10 * verts[1].x + w11 * verts[2].x + w01 * verts[3].x,
                w00 * verts[0].y + w10 * verts[1].y + w11 * verts[2].y + w01 * verts[3].y,
            };

            // Lerp for colors
            if (colors) {
                meshColors[idx] = {
                    w00 * colors[0].r + w10 * colors[1].r + w11 * colors[2].r + w01 * colors[3].r,
                    w00 * colors[0].g + w10 * colors[1].g + w11 * colors[2].g + w01 * colors[3].g,
                    w00 * colors[0].b + w10 * colors[1].b + w11 * colors[2].b + w01 * colors[3].b,
                    w00 * colors[0].a + w10 * colors[1].a + w11 * colors[2].a + w01 * colors[3].a,
                };
            }

            // Lerp for texture coordinates
            if (texs) {
                meshTexs[idx] = {
                    w00 * texs[0].x + w10 * texs[1].x + w11 * texs[2].x + w01 * texs[3].x,
                    w00 * texs[0].y + w10 * texs[1].y + w11 * texs[2].y + w01 * texs[3].y,
                };
            }
        }
    }

    // Triangle indices (2 triangles per sub-quad)
    int triCount = n * n * 2;
    std::vector<int> indices(triCount * 3);
    int idx = 0;

    for (int row = 0; row < n; row++) {
        for (int col = 0; col < n; col++) {
            // Corner indices of sub-quad
            int tl = row * (n + 1) + col;
            int tr = tl + 1;
            int bl = tl + (n + 1);
            int br = bl + 1;

            // Upper-left triangle
            indices[idx++] = tl;
            indices[idx++] = tr;
            indices[idx++] = bl;

            // Lower-right triangle
            indices[idx++] = tr;
            indices[idx++] = br;
            indices[idx++] = bl;
        }
    }

    // Draw the mesh
    drawMesh(meshVerts.data(),
             colors ? meshColors.data() : nullptr,
             texs ? meshTexs.data() : nullptr,
             triCount, indices.data(), paint);
}

void WMZCanvas::save() {
    fCTMStack.push_back(fCTM);
}

void WMZCanvas::restore() {
    fCTM = fCTMStack.back();
    fCTMStack.pop_back();
}

void WMZCanvas::concat(const GMatrix& matrix) {
    fCTM = GMatrix::Concat(fCTM, matrix);
}

std::unique_ptr<GCanvas> GCreateCanvas(const GBitmap& device) {
    return std::unique_ptr<GCanvas>(new WMZCanvas(device));
}

std::string GDrawSomething(GCanvas* canvas, GISize dim) {
    float w = (float)dim.width;
    float h = (float)dim.height;

    // Sunset gradient
    const GColor sky[] = {
        {1.0f, 0.45f, 0.0f, 1},
        {1.0f, 0.4f, 0.5f, 1}, 
        {0.3f, 0.15f, 0.35f, 1},
    };
    canvas->drawRect(GRect::WH(w, h), GPaint(GShader::LinearGradient({0,0}, {0,h}, sky, 3)));

    // Sun
    float cx = w * 0.5f, cy = h * 0.75f, r = 40;
    const int N = 40;
    GPoint sun[N];
    for (int i = 0; i < N; i++) {
        float t = 2 * gFloatPI * i / N;
        sun[i] = {cx + r * cosf(t), cy + r * sinf(t)};
    }
    canvas->drawConvexPolygon(sun, N, GPaint({1.0f, 0.9f, 0.3f, 1}));

    // Black ground
    canvas->drawRect(GRect::LTRB(0, h * 0.75f, w, h), GPaint({0, 0, 0, 1}));

    return "Ahh the sun";
}