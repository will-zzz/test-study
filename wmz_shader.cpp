/*
 *  Copyright 2026 Will Zakielarz
 */

#include "include/GShader.h"
#include "include/GColor.h"
#include "include/GBitmap.h"
#include "include/GMatrix.h"
#include "include/GPoint.h"
#include "include/GPixel.h"
#include "include/GMath.h"
#include <algorithm>
#include <cmath>

// Helper: Pin to tile
static int pin_to_tile(int v, int limit, GTileMode tm) {
    if (limit <= 0) {
        return 0;
    }

    // Map [0, limit - 1] based on tile mode
    switch (tm) {
        case GTileMode::kClamp:
            // Clamp to edges
            return std::max(0, std::min(limit - 1, v));
        case GTileMode::kRepeat: {
            int m = v % limit;
            return m < 0 ? m + limit : m;
        }
        case GTileMode::kMirror: {
            // Bounce back every other period
            if (limit == 1) {
                return 0;
            }
            int period = 2 * limit;
            int m = v % period;
            if (m < 0) {
                m += period;
            }
            if (m >= limit) {
                m = period - 1 - m;
            }
            return m;
        }
    }
    return 0;
}

static float pin_unit_to_tile(float t, GTileMode tm) {
    // Map continuous gradient to [0, 1]
    switch (tm) {
        case GTileMode::kClamp:
            return std::max(0.0f, std::min(1.0f, t));
        case GTileMode::kRepeat:
            // Keep fractional part
            return t - std::floor(t);
        case GTileMode::kMirror: {
            // Reflect over odd interval
            float m = std::fmod(t, 2.0f);
            if (m < 0) {
                m += 2.0f;
            }
            if (m > 1.0f) {
                m = 2.0f - m;
            }
            return m;
        }
    }
    return t;
}

class BitmapShader : public GShader {
public:
    // Constructor
    BitmapShader(const GBitmap& bitmap, const GMatrix& localMatrix, GTileMode tm)
        : fBitmap(bitmap)
        , fLocalMatrix(localMatrix)
        , fTileMode(tm)
    {}

    bool isOpaque() override {
        return fBitmap.isOpaque();
    }

    std::unique_ptr<Context> makeContext(const GMatrix& ctm) override {
        // Combine CTM with local matrix (bitmap -> device)
        GMatrix total = GMatrix::Concat(ctm, fLocalMatrix);
        // Invert so we can sample (device -> bitmap)
        auto inv = total.invert();
        if (inv) {
            return std::make_unique<BitmapContext>(fBitmap, *inv, fTileMode);
        }
        return nullptr;
    }

private:
    GBitmap fBitmap;
    GMatrix fLocalMatrix;
    GTileMode fTileMode;

    class BitmapContext : public Context {
    public:
        BitmapContext(const GBitmap& bitmap, const GMatrix& inv, GTileMode tm)
            : fBitmap(bitmap)
            , fInverse(inv)
            , fTileMode(tm)
        {}

        void shadeRow(int x, int y, int count, GPixel row[]) override {
            // Map first pixel center
            GPoint loc = fInverse * GPoint{(float)x + 0.5f, (float)y + 0.5f};
            // Bitmap space when we advance 1 pixel
            GVector step = fInverse.e0();
            int w = fBitmap.width();
            int h = fBitmap.height();

            for (int i = 0; i < count; i++) {
                int px = GFloorToInt(loc.x);
                int py = GFloorToInt(loc.y);
                px = pin_to_tile(px, w, fTileMode);
                py = pin_to_tile(py, h, fTileMode);
                row[i] = *fBitmap.getAddr(px, py);
                // Next pixel's bitmap coords
                loc.x += step.x;
                loc.y += step.y;
            }
        }

    private:
        GBitmap fBitmap;
        GMatrix fInverse;
        GTileMode fTileMode;
    };
};

std::shared_ptr<GShader> GShader::Bitmap(const GBitmap& bitmap, const GMatrix& localMatrix, GTileMode tm) {
    return std::make_shared<BitmapShader>(bitmap, localMatrix, tm);
}

class LinearGradientShader : public GShader {
public:
    // Constructor
    LinearGradientShader(GPoint p0, GPoint p1, const GColor colors[], int count, GTileMode tm)
        : fColors(colors, colors + count)
        , fTileMode(tm)
    {
        // Matrix that maps p0 -> p1 onto the unit x-axis
        GVector dp = {p1.x - p0.x, p1.y - p0.y};
        fLocalMatrix = GMatrix(dp.x, -dp.y, p0.x,
                               dp.y,  dp.x, p0.y);
    }

    bool isOpaque() override {
        for (const auto& c : fColors) {
            if (c.a < 1.0f) return false;
        }
        return true;
    }

    std::unique_ptr<Context> makeContext(const GMatrix& ctm) override {
        // Combine CTM with local matrix (gradient -> device)
        GMatrix total = GMatrix::Concat(ctm, fLocalMatrix);
        // Invert so we can sample (device -> gradient t value)
        auto inv = total.invert();
        if (!inv) {
            return nullptr;
        }
        return std::make_unique<GradientContext>(fColors, *inv, fTileMode);
    }

private:
    std::vector<GColor> fColors;
    GMatrix fLocalMatrix;
    GTileMode fTileMode;

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

        return GPixel_PackARGB(alpha, red, green, blue);
    }

    class GradientContext : public Context {
    public:
        GradientContext(const std::vector<GColor>& colors, const GMatrix& inv, GTileMode tm)
            : fColors(colors)
            , fInverse(inv)
            , fTileMode(tm)
        {}

        void shadeRow(int x, int y, int count, GPixel row[]) override {
            // Map first pixel center
            GPoint loc = fInverse * GPoint{(float)x + 0.5f, (float)y + 0.5f};
            // Gradient space when we advance 1 pixel
            GVector step = fInverse.e0();
            int n = (int)fColors.size();

            for (int i = 0; i < count; i++) {
                float t = pin_unit_to_tile(loc.x, fTileMode);

                // Figure out which two colors we're between
                float scaled = t * (n - 1);
                int idx = GFloorToInt(scaled);
                if (idx < 0) {
                    idx = 0;
                }
                if (idx >= n - 1) {
                    idx = n - 2;
                }
                float frac = scaled - idx;

                // Lerp between the two colors
                GColor c;
                c.r = fColors[idx].r + frac * (fColors[idx+1].r - fColors[idx].r);
                c.g = fColors[idx].g + frac * (fColors[idx+1].g - fColors[idx].g);
                c.b = fColors[idx].b + frac * (fColors[idx+1].b - fColors[idx].b);
                c.a = fColors[idx].a + frac * (fColors[idx+1].a - fColors[idx].a);

                row[i] = color_to_pixel(c);
                // Next pixel's gradient coords
                loc.x += step.x;
                loc.y += step.y;
            }
        }

    private:
        std::vector<GColor> fColors;
        GMatrix fInverse;
        GTileMode fTileMode;
    };
};

std::shared_ptr<GShader> GShader::LinearGradient(GPoint p0, GPoint p1, const GColor colors[], int count, GTileMode tm) {
    if (count < 1) return nullptr;
    return std::make_shared<LinearGradientShader>(p0, p1, colors, count, tm);
}

class ModAlphaShader : public GShader {
public:
    ModAlphaShader(std::shared_ptr<GShader> base, float alphaScale)
        : fBase(std::move(base))
        // Clamp to [0,1]
        , fAlphaScale(std::max(0.0f, std::min(1.0f, alphaScale)))
    {}

    bool isOpaque() override {
        // Opaque if base is opaque and scale keeps alpha unchanged
        return fAlphaScale >= 1.0f && fBase->isOpaque();
    }

    std::unique_ptr<Context> makeContext(const GMatrix& ctm) override {
        auto baseCtx = fBase->makeContext(ctm);
        if (!baseCtx) {
            return nullptr;
        }
        return std::make_unique<ModAlphaContext>(std::move(baseCtx), fAlphaScale);
    }

private:
    std::shared_ptr<GShader> fBase;
    float fAlphaScale;

    class ModAlphaContext : public Context {
    public:
        ModAlphaContext(std::unique_ptr<Context> base, float alphaScale)
            : fBase(std::move(base))
            , fAlphaScale(alphaScale)
        {}

        void shadeRow(int x, int y, int count, GPixel row[]) override {
            // First get premul pixels
            fBase->shadeRow(x, y, count, row);
            if (fAlphaScale >= 1.0f) {
                return;
            }

            // Pixels are premul, scale all channels by same alpha factor
            for (int i = 0; i < count; ++i) {
                const int a = GRoundToInt(GPixel_GetA(row[i]) * fAlphaScale);
                const int r = GRoundToInt(GPixel_GetR(row[i]) * fAlphaScale);
                const int g = GRoundToInt(GPixel_GetG(row[i]) * fAlphaScale);
                const int b = GRoundToInt(GPixel_GetB(row[i]) * fAlphaScale);
                row[i] = GPixel_PackARGB(a, r, g, b);
            }
        }

    private:
        std::unique_ptr<Context> fBase;
        float fAlphaScale;
    };
};

std::shared_ptr<GShader> GShader::ModAlpha(std::shared_ptr<GShader> base, float alphaScale) {
    if (!base) {
        return nullptr;
    }
    return std::make_shared<ModAlphaShader>(std::move(base), alphaScale);
}

// Shared blend helper
#include "wmz_blend.h"

// Blends outputs of two shaders using blend mode
class BlendShader : public GShader {
public:
    BlendShader(std::shared_ptr<GShader> src, std::shared_ptr<GShader> dst, GBlendMode mode)
    : fSrc(std::move(src)), fDst(std::move(dst)), fMode(mode) {}

    bool isOpaque() override {
        return false;
    }

    std::unique_ptr<Context> makeContext(const GMatrix& ctm) override {
        // Get contexts for both shaders
        auto srcCtx = fSrc->makeContext(ctm);
        auto dstCtx = fDst->makeContext(ctm);

        if (!srcCtx || !dstCtx) return nullptr;

        return std::make_unique<BlendCtx>(std::move(srcCtx), std::move(dstCtx), fMode);
    }

private:
    std::shared_ptr<GShader> fSrc, fDst;
    GBlendMode fMode;

    class BlendCtx : public Context {
    public:
        BlendCtx(std::unique_ptr<Context> src, std::unique_ptr<Context> dst, GBlendMode mode)
        : fSrc(std::move(src)), fDst(std::move(dst)), fMode(mode) {}

        void shadeRow(int x, int y, int count, GPixel row[]) override {
            // Get pixels from both shaders
            std::vector<GPixel> dstRow(count);
            fSrc->shadeRow(x, y, count, row);
            fDst->shadeRow(x, y, count, dstRow.data());

            // Blend each pixel
            for (int i = 0; i < count; i++) {
                row[i] = apply_blend(row[i], dstRow[i], fMode);
            }
        }

    private:
        std::unique_ptr<Context> fSrc, fDst;
        GBlendMode fMode;
    };
};

std::shared_ptr<GShader> GShader::Blend(std::shared_ptr<GShader> src, std::shared_ptr<GShader> dst, GBlendMode mode) {
    if (!src || !dst) return nullptr;
    
    return std::make_shared<BlendShader>(std::move(src), std::move(dst), mode);
}
