#pragma once

#include "glm/glm.hpp"
#include <vector>
#include <string>

struct TriangleMesh {
	std::vector<glm::vec3> vertex;
	std::vector<glm::vec3> normal;
	std::vector<glm::vec2> texcoord;
	std::vector<glm::ivec3> index;

	// Material Properties
	glm::vec3 diffuse;
};

struct Model {
	~Model() 
	{ for(auto mesh: meshes) delete mesh; }

	std::vector<TriangleMesh*> meshes;
	// ! Bounding box of all vertices in the model
	glm::vec3 boundsMin;
	glm::vec3 boundsMax;
	glm::vec3 boundsCenter;
	glm::vec3 boundsSpan;
};


Model* loadOBJ(const std::string& objFile);