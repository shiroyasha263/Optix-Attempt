#include "SampleRenderer.h"

// our helper library for window handling
#include "glfWindow/GLFWindow.h"
#include <GL/gl.h>

#include "glm/glm.hpp"

struct SampleWindow : public osc::GLFCameraWindow {
    SampleWindow(const std::string& title,
                 const Model* model,
                 const Camera& camera,
                 const float worldScale)
        : GLFCameraWindow(title, camera.from, camera.at, camera.up, worldScale), sample(model) {}

    virtual void render() override {
        if (cameraFrame.modified) {
            sample.setCamera(Camera{ cameraFrame.get_from(),
                                     cameraFrame.get_at(),
                                     cameraFrame.get_up() });
            cameraFrame.modified = false;
        }
        sample.render();
    }

    virtual void draw() override {
        sample.downloadPixels(pixels.data());
        if (fbTexture == 0)
            glGenTextures(1, &fbTexture);

        glBindTexture(GL_TEXTURE_2D, fbTexture);
        GLenum texFormat = GL_RGBA;
        GLenum texelType = GL_UNSIGNED_BYTE;
        glTexImage2D(GL_TEXTURE_2D, 0, texFormat, fbSize.x, fbSize.y, 0, GL_RGBA,
            texelType, pixels.data());

        glDisable(GL_LIGHTING);
        glColor3f(1, 1, 1);

        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();

        glEnable(GL_TEXTURE_2D);
        glBindTexture(GL_TEXTURE_2D, fbTexture);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        glDisable(GL_DEPTH_TEST);

        glViewport(0, 0, fbSize.x, fbSize.y);

        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glOrtho(0.f, (float)fbSize.x, 0.f, (float)fbSize.y, -1.f, 1.f);

        glBegin(GL_QUADS);
        {
            glTexCoord2f(0.f, 0.f);
            glVertex3f(0.f, 0.f, 0.f);

            glTexCoord2f(0.f, 1.f);
            glVertex3f(0.f, (float)fbSize.y, 0.f);

            glTexCoord2f(1.f, 1.f);
            glVertex3f((float)fbSize.x, (float)fbSize.y, 0.f);

            glTexCoord2f(1.f, 0.f);
            glVertex3f((float)fbSize.x, 0.f, 0.f);
        }
        glEnd();
    }

    virtual void resize(const glm::ivec2& newSize) {
        fbSize = newSize;
        sample.resize(newSize);
        pixels.resize(newSize.x * newSize.y);
    }

    glm::ivec2            fbSize;
    GLuint                fbTexture{ 0 };
    SampleRenderer        sample;
    std::vector<uint32_t> pixels;
};

inline uint32_t XOrShift32(uint32_t* state)
{
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

/*! main entry point to this example - initially optix, print hello
  world, then exit */
extern "C" int main(int ac, char** av) {
    try {
        Model* model = loadOBJ("C:/Users/Vishu.Main-Laptop/Downloads/optix-examples-main/models/CornellBox/CornellBox-Water.obj");
        
        std::cout << "Model loaded perfectly!\n";
        
        Camera camera = { /*from*/glm::vec3(model->boundsCenter) + glm::vec3(2.f),
                        /* at */glm::vec3(model->boundsCenter),
                        /* up */glm::vec3(0.f,1.f,0.f) };
        
        // something approximating the scale of the world, so the
        // camera knows how much to move for any given user interaction:
        const float worldScale = glm::length(model->boundsSpan);
        
        SampleWindow* window = new SampleWindow("Optix 7 Course Example",
                                                model, camera, worldScale);
        window->run();

    }
    catch (std::runtime_error& e) {
        std::cout << TERMINAL_RED << "FATAL ERROR: " << e.what()
            << TERMINAL_DEFAULT << std::endl;
        exit(1);
    }
    return 0;
}