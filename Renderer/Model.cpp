#include "Model.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include "3rdParty/tiny_obj_loader.h"

#define STB_IMAGE_IMPLEMENTATION
#include "3rdParty/stb_image.h"

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

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

	// just for sanity's sake:
	if (mesh->texcoord.size() > 0)
		mesh->texcoord.resize(mesh->vertex.size());
	// just for sanity's sake:
	if (mesh->normal.size() > 0)
		mesh->normal.resize(mesh->vertex.size());

	return newID;
}

/*! load a texture (if not already loaded), and return its ID in the
	model's textures[] vector. Textures that could not get loaded
	return -1 */
int loadTexture(Model* model,
				std::map<std::string, int> &knownTextures,
				const std::string &inFileName,
				const std::string &modelPath) {
	
	// If the input file is empty send this
	if (inFileName == "")
		return -1;

	// Check if the file is already loaded, and if so return the file index
	if (knownTextures.find(inFileName) != knownTextures.end())
		return knownTextures[inFileName];

	// Fix any file name issues and get the exact file path
	std::string fileName = inFileName;
	for (auto& c : fileName)
		if (c == '\\') c = '/';
	fileName = modelPath + "/" + fileName;

	// Get the image and its resolution
	glm::ivec2 res;
	int comp;

	// STBI has a habit of inversing images
	stbi_set_flip_vertically_on_load(true);
	unsigned char* image = stbi_load(fileName.c_str(), &res.x, &res.y, &comp, STBI_rgb_alpha);
	stbi_set_flip_vertically_on_load(false);

	int textureID = -1;
	if (image) {
		textureID = (int)model->textures.size();
		Texture* texture = new Texture;
		texture->resolution = res;
		texture->pixel = (uint32_t*)image;

		model->textures.push_back(texture);
	}
	else {
		std::cout << "Could not load texture from " << fileName << "!\n";
	}

	knownTextures[inFileName] = textureID;
	return textureID;
}

Model* loadOBJ(const std::string& objFile) {
	Model* model = new Model;

	// Check if there is a mtlDirectory
	const std::string modelDir
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
							modelDir.c_str(),
							/*triangulate*/ true);

	// Read error handling
	if (!readOK)
		throw std::runtime_error("Could not read OBJ Model from " + objFile + ": " + modelDir + " : " + err);
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
		// Get a filename to int mapping
		std::map<std::string, int> knownTexture;

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
				mesh->emmissive = 1.f * (const glm::vec3&)materials[materialID].emission;
				mesh->diffuseTextureID = loadTexture(model, knownTexture, materials[materialID].diffuse_texname, modelDir);
				mesh->specular = (const glm::vec3&)materials[materialID].specular;
				mesh->shininess = materials[materialID].shininess;
				mesh->ior = materials[materialID].ior;
				mesh->illum = materials[materialID].illum;
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
	std::cout << "Loaded " << model->textures.size() << " textures" << std::endl;
	return model;
}

TriangleMesh* processMesh(Model* model, aiMesh* mesh, const aiScene* scene, std::string modelDir) {
	TriangleMesh* triMesh = new TriangleMesh;

	for (int idx = 0; idx < mesh->mNumVertices; idx++) {
		triMesh->vertex.push_back(glm::vec3(
			mesh->mVertices[idx].x, mesh->mVertices[idx].y, mesh->mVertices[idx].z));

		if (mesh->HasNormals())
			triMesh->normal.push_back(glm::vec3(
				mesh->mNormals[idx].x, mesh->mNormals[idx].y, mesh->mNormals[idx].z));

		if (mesh->mTextureCoords[0])
			triMesh->texcoord.push_back(glm::vec2(
				mesh->mTextureCoords[0][idx].x, mesh->mTextureCoords[0][idx].y));
	}

	// UPDATE NEEDED!!!!!!!!!!!!!
	// ONLY WORKS FOR TRIANGLESSS
	for (int i = 0; i < mesh->mNumFaces; i++) {
		aiFace face = mesh->mFaces[i];
		for (unsigned int j = 0; j < face.mNumIndices; j = j + 3)
			triMesh->index.push_back(glm::ivec3(face.mIndices[j], face.mIndices[j + 1], face.mIndices[j + 2]));
	}

	// Get a filename to int mapping
	std::map<std::string, int> knownTexture;

	if (mesh->mMaterialIndex >= 0) {
		aiString str;
		aiMaterial* mtl = scene->mMaterials[mesh->mMaterialIndex];
		mtl->GetTexture(aiTextureType_DIFFUSE, 0, &str);
		
		aiColor4D getColor;
		float getFloat;

		// std::cout << std::endl;

		if (AI_SUCCESS == aiGetMaterialColor(mtl, AI_MATKEY_COLOR_DIFFUSE, &getColor)) {
			triMesh->diffuse = glm::vec3(getColor.r, getColor.g, getColor.b);
			// std::cout << mtl->GetName().C_Str() << " Diffuse R: " << getColor.r << ", G: " << getColor.g << ", B: " << getColor.b << std::endl;
		}
		else
			triMesh->diffuse = glm::vec3(1.f, 1.f, 1.f);

		if (AI_SUCCESS == aiGetMaterialColor(mtl, AI_MATKEY_COLOR_EMISSIVE, &getColor)) {
			triMesh->emmissive = 10.f * glm::vec3(getColor.r, getColor.g, getColor.b);
			// std::cout << mtl->GetName().C_Str() << " Emmissive R: " << getColor.r << ", G: " << getColor.g << ", B: " << getColor.b << std::endl;
		}
		else
			triMesh->emmissive = glm::vec3(0.f, 0.f, 0.f);

		if (AI_SUCCESS == aiGetMaterialColor(mtl, AI_MATKEY_COLOR_SPECULAR, &getColor)) {
			triMesh->specular = 1.f * glm::vec3(getColor.r, getColor.g, getColor.b);
			// std::cout << mtl->GetName().C_Str() << " Emmissive R: " << getColor.r << ", G: " << getColor.g << ", B: " << getColor.b << std::endl;
		}
		else
			triMesh->specular = glm::vec3(0.f, 0.f, 0.f);

		if (AI_SUCCESS == aiGetMaterialFloat(mtl, AI_MATKEY_SHININESS, &getFloat)) {
			triMesh->shininess = getFloat;
			// std::cout << mtl->GetName().C_Str() << " shininess: " << getFloat << std::endl;
		}
		else
			triMesh->shininess = 10.f;

		// std::cout << std::endl;
		std::string texname = str.C_Str();
		texname = texname;
		triMesh->diffuseTextureID = loadTexture(model, knownTexture, texname, modelDir);
	}

	return triMesh;
}

void processNode(Model* model, aiNode* node, const aiScene* scene, std::string modelDir)
{
	// process all the node's meshes (if any)
	for (unsigned int i = 0; i < node->mNumMeshes; i++)
	{
		aiMesh* mesh = scene->mMeshes[node->mMeshes[i]];
		model->meshes.push_back(processMesh(model, mesh, scene, modelDir));
	}
	// then do the same for each of its children
	for (unsigned int i = 0; i < node->mNumChildren; i++)
	{
		processNode(model, node->mChildren[i], scene, modelDir);
	}
}

Model* loadModel(const std::string& modelFile) {
	Model* model = new Model;

	Assimp::Importer import;
	const aiScene * scene = import.ReadFile(modelFile, aiProcess_Triangulate);

	if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode)
	{
		throw std::runtime_error("ERROR::ASSIMP");
	}
	std::string modelDir = modelFile.substr(0, modelFile.find_last_of('/'));

	std::cout << "Loading Model Using ASSIMP\n";

	processNode(model, scene->mRootNode, scene, modelDir);

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
	std::cout << "Loaded " << model->textures.size() << " textures" << std::endl;

	return model;
}