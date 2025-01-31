#include "Model.h"
#define TINYOBJLOADER_IMPLEMENTATION

#include "3rdParty/tiny_obj_loader.h"

#include <iostream>
#include <set>
#include <limits>

namespace std {
	inline bool operator<(const tinyobj::index_t& a,
		const tinyobj::index_t& b) {
		if (a.vertex_index < b.vertex_index) return true;
		if (a.vertex_index > b.vertex_index) return false;

		if (a.normal_index < b.normal_index) return true;
		if (a.normal_index > b.normal_index) return false;

		if (a.texcoord_index < b.texcoord_index) return true;
		if (a.texcoord_index > b.texcoord_index) return false;

		return false;
	}
}

glm::vec3 randomColor(uint32_t i) {
	int r = i * 13 * 17 + 0x234235;
	int g = i * 7 * 3 * 5 + 0x773477;
	int b = i * 11 * 19 + 0x223766;
	return glm::vec3((r & 255) / 255.f,
		(g & 255) / 255.f,
		(b & 255) / 255.f);
}


/*! find vertex with given position, normal, texcoord, and return
	its vertex ID, or, if it doesn't exit, add it to the mesh, and
	its just-created index */
int addVertex(TriangleMesh *mesh,
			  tinyobj::attrib_t &attributes,
			  const tinyobj::index_t &idx,
			  std::map<tinyobj::index_t, int> &knownVertices) {
	
	// If idx is a part of knownVertices in its complete length anywhere
	if (knownVertices.find(idx) != knownVertices.end())
		return knownVertices[idx]; // If it is, return the vertex index at idx

	const glm::vec3* vertex_array = (const glm::vec3*)attributes.vertices.data();
	const glm::vec3* normal_array = (const glm::vec3*)attributes.normals.data();
	const glm::vec2* texcoord_array = (const glm::vec2*)attributes.texcoords.data();

	int newID = mesh->vertex.size();
	knownVertices[idx] = newID;

	mesh->vertex.push_back(vertex_array[idx.vertex_index]);

	// If normals exist in this data, add normals for each point
	// on the vertex, in this case it seems to add the same normal to all vertices
	if (idx.normal_index >= 0) {
		while (mesh->normal.size() < mesh->vertex.size())
			mesh->normal.push_back(normal_array[idx.normal_index]);
	}

	// If texcoords exist in this data, add texcoords for each point
	// on the vertex, in this case it seems to add the same texcoords to all vertices
	if (idx.texcoord_index >= 0) {
		while (mesh->texcoord.size() < mesh->vertex.size())
			mesh->texcoord.push_back(texcoord_array[idx.texcoord_index]);
	}

	return newID;
}

Model* loadOBJ(const std::string& objFile) {
	Model* model = new Model;

	// Check if there is a mtlDirectory
	const std::string mtlDir
		= objFile.substr(0, objFile.rfind('/') + 1);

	// These are the things that we will get from our files
	tinyobj::attrib_t attributes;
	std::vector<tinyobj::shape_t> shapes;
	std::vector<tinyobj::material_t> materials;
	std::string err = "";

	// Check if the read went well, and fill the variables
	bool readOK
		= tinyobj::LoadObj(	&attributes,
							&shapes,
							&materials,
							&err,
							&err,
							objFile.c_str(),
							mtlDir.c_str(),
							/*triangulate*/ true);

	// Read error handling
	if (!readOK)
		throw std::runtime_error("Could not read OBJ Model from " + objFile + ": " + mtlDir + " : " + err);
	if (materials.empty())
		throw std::runtime_error("Could not parse materials. . . . . ");

	// Read went well
	std::cout << "Done loading obj file - Found " << shapes.size() << "shapes with " << materials.size() << "materials\n";

	//// Now to fill our Model with meshes!
	// For every shape rerun the loop
	for (int shapeID = 0; shapeID < (int)shapes.size(); shapeID++) {
		tinyobj::shape_t &shape = shapes[shapeID];

		// Get a list of material IDs that are being used in this shape
		std::set<int> materialIDs;
		for (auto faceMatID : shape.mesh.material_ids)
			materialIDs.insert(faceMatID);

		// Get a index object to int mapping
		std::map<tinyobj::index_t, int> knownVertices;

		// For every material ID loop over
		for (auto materialID : materialIDs) {
			TriangleMesh* mesh = new TriangleMesh;
			
			// for every face check if this face uses the material id being used
			// if not check the next face!
			for (int faceID = 0; faceID < shape.mesh.material_ids.size(); faceID++) {
				if (shape.mesh.material_ids[faceID] != materialID) continue;
				// Get the mesh index from the shape using the faceID
				tinyobj::index_t idx_0 = shape.mesh.indices[3 * faceID + 0];
				tinyobj::index_t idx_1 = shape.mesh.indices[3 * faceID + 1];
				tinyobj::index_t idx_2 = shape.mesh.indices[3 * faceID + 2];

				// Add the different kind of attribute datas to the mesh
				// i.e. vertex, normal, texcoords
				// and pass the idx values of this triangle or primary shape
				// to our mesh!
				glm::ivec3 idx(
					addVertex(mesh, attributes, idx_0, knownVertices),
					addVertex(mesh, attributes, idx_1, knownVertices),
					addVertex(mesh, attributes, idx_2, knownVertices));
				mesh->index.push_back(idx);
				// Anything with the same material ID is given the same diffuse color
				mesh->diffuse = (const glm::vec3&)materials[materialID].diffuse;
				mesh->diffuse = randomColor(materialID);
			}

			if (mesh->vertex.empty())
				delete mesh;
			else
				model->meshes.push_back(mesh);
		}
	}

	// of course, you should be using tbb::parallel_for for stuff
	// like this:
	model->boundsMin = glm::vec3(std::numeric_limits<float>::max());
	model->boundsMax = glm::vec3(-std::numeric_limits<float>::max());

	for (auto mesh : model->meshes) {
		for (auto vtx : mesh->vertex) {
			model->boundsMin = glm::min(model->boundsMin, vtx);
			model->boundsMax = glm::max(model->boundsMax, vtx);
		}
	}

	model->boundsCenter = model->boundsMin + (model->boundsMax - model->boundsMin) * .5f;
	model->boundsSpan = model->boundsMax - model->boundsMin;

	std::cout << "created a total of " << model->meshes.size() << " meshes" << std::endl;
	return model;
}