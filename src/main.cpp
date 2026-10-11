#ifdef __APPLE__
// For portability subset device extension, required for MoltenVK
#define VK_ENABLE_BETA_EXTENSIONS
#endif

#if defined(__INTELLISENSE__) || !defined(USE_CPP20_MODULES)

#include <vulkan/vulkan_raii.hpp>
#include <vulkan/vulkan.h>

#else
import vulkan_hpp;
#endif

#ifdef _WIN32
#define VK_USE_PLATFORM_WIN32_KHR
#define GLFW_EXPOSE_NATIVE_WIN32
#pragma message ("WIN")
#elif defined(__APPLE__) || defined(__MACH__)
#define VK_USE_PLATFORM_METAL_EXT
#define GLFW_EXPOSE_NATIVE_COCOA
#pragma message ("Apple")
#else
#pragma message ("None")
#endif

#define GLFW_INCLUDE_VULKAN

#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>
// automatic alignment for glm structs
// Note: this breaks down in nested structs. Need explicit alignas()
#define GLM_FORCE_DEFAULT_ALIGNED_GENTYPES
#include <glm/glm.hpp>

#include <iostream>
#include <map>
#include <stdexcept>
#include <cstdlib>
#include <fstream>
// Enable FP exceptions
#pragma STDC FENV_ACCESS ON
#include <cfenv>

#include "solver.h"

constexpr uint32_t WIDTH = 800;
constexpr uint32_t HEIGHT = 600;
constexpr uint32_t NUM_CELLS = 100;

constexpr int MAX_FRAMES_IN_FLIGHT = 2;

const std::vector<char const *> validationLayers = {
        "VK_LAYER_KHRONOS_validation",
};

#ifdef NDEBUG
constexpr bool enableValidationLayers = false;
#else
constexpr bool enableValidationLayers = true;
#endif

// VAO
struct Vertex {
    glm::vec2 pos;
    glm::vec2 texCoord;

    static vk::VertexInputBindingDescription getBindingDescription() {
        return {
            .binding = 0,
            .stride = sizeof(Vertex),
            .inputRate = vk::VertexInputRate::eVertex
        };
    }

    static std::array<vk::VertexInputAttributeDescription, 2> getAttributeDescriptions() {
        return {{{.location = 0, .binding = 0, .format = vk::Format::eR32G32Sfloat, .offset = offsetof(Vertex, pos)},
                 {.location = 1, .binding = 0, .format = vk::Format::eR32G32Sfloat, .offset = offsetof(Vertex, texCoord)}}};
    }
};

struct UniformBufferObject {
    glm::mat4 model;
    glm::mat4 view;
    glm::mat4 proj;
};

// [-1, 1]
const std::vector<Vertex> vertices = {
        {{-1.0f, -1.0f}, {1.0f, 0.0f}},
        {{1.0f, -1.0f}, {0.0f, 0.0f}},
        {{1.0f, 1.0f}, {0.0f, 1.0f}},
        {{-1.0f, 1.0f}, {1.0f, 1.0f}}
};

const std::vector<uint16_t> indices = {0, 1, 2, 2, 3, 0};

template<std::floating_point T=double, SOLVER_TYPE S = SOLVER_TYPE::Eulerian2D>
class FluidSimApplication {
    // should attempt change to consteval switch case if more fluid sims are created
//    using SOLVER_T = std::conditional_t<S == SOLVER_TYPE::Eulerian2D, Eulerian2D<T>, SolverBase>;
    using SOLVER_T = Eulerian2D<double>;
public:
    explicit FluidSimApplication(int i = NUM_CELLS, int j = NUM_CELLS) : fluidSolver(SOLVER_T(i, j)), texWidth(j), texHeight(i) {
        std::cerr << std::format("i: {}, j: {}\n", i,j);
    };

    void run() {
        initWindow();
        initVulkan();
        mainLoop();
        cleanup();
    }

private:
    void initWindow() {
        glfwInit();
        // Not OpenGL hint
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

        window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
    }

    void initVulkan() {
        createInstance();
        setupDebugMessenger();
        createSurface();
        pickPhysicalDevice();
        createLogicalDevice();
        createSwapChain();
        createImageViews();
        createDescriptorSetLayout();
        createGraphicsPipeline();
        createCommandPool();
        // texture mapping
        createTextureImages();
        createTextureImageViews();
        createTextureSampler();
        // buffers
        createVertexBuffer();
        createIndexBuffer();
        createUniformBuffers();
        createDescriptorPool();
        createDescriptorSets();
        createCommandBuffers();
        createSyncObjects();
    }

    void createSurface() {
        VkSurfaceKHR _surface;
        // cross-platform window surface creation
        if (glfwCreateWindowSurface(*instance, window, nullptr, &_surface) != 0) {
            throw std::runtime_error("failed to create window surface!");
        }
        surface = vk::raii::SurfaceKHR(instance, _surface);
    }

    void setupDebugMessenger() {
        if (!enableValidationLayers) return; // NOLINT

        vk::DebugUtilsMessageSeverityFlagsEXT severityFlags(vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning |
                                                            vk::DebugUtilsMessageSeverityFlagBitsEXT::eError);
        vk::DebugUtilsMessageTypeFlagsEXT messageTypeFlags(
                vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral | vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance |
                vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation);
        vk::DebugUtilsMessengerCreateInfoEXT debugUtilsMessengerCreateInfoExt{.messageSeverity = severityFlags,
                .messageType = messageTypeFlags,
                .pfnUserCallback = &debugCallback};
        debugMessenger = instance.createDebugUtilsMessengerEXT(debugUtilsMessengerCreateInfoExt);
    }

    void mainLoop() {
        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();
            try {
                fluidSolver.iterate();
            } catch (std::exception &e) {
                std::cerr << e.what() << std::endl;

            }
            drawFrame();
        }
    }

    void cleanup() {
        glfwDestroyWindow(window);

        glfwTerminate();
    }

    // TODO: swapchain recreation in window resize. https://docs.vulkan.org/tutorial/latest/03_Drawing_a_triangle/04_Swap_chain_recreation.html

    void createInstance() {
        constexpr vk::ApplicationInfo appInfo{.pApplicationName = "FluidSimApp",
                .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
                .pEngineName = "No Engine",
                .engineVersion = VK_MAKE_VERSION(1, 0, 0),
                .apiVersion = vk::ApiVersion14
        };

        // Get required instance extensions from GLFW
        auto requiredExtensions = getRequiredInstanceExtensions();
        // Required MoltenVK extension
        // Keeping it like this instead of direct PB for flexibility
        std::vector<const char *> requiredVulkanExtensions = {
                vk::KHRPortabilityEnumerationExtensionName
        };
        requiredExtensions.insert(requiredExtensions.end(), requiredVulkanExtensions.begin(),
                                  requiredVulkanExtensions.end());

        // Check if the required GLFW extensions are supported by the Vulkan implementation
        auto extensionProperties = context.enumerateInstanceExtensionProperties();
        auto unsupportedPropertyIt = std::ranges::find_if(requiredExtensions,
                                                          [&extensionProperties](auto const &requiredExtension) {
                                                              return std::ranges::none_of(extensionProperties,
                                                                                          [requiredExtension](
                                                                                                  auto const &extensionProperty) {
                                                                                              return strcmp(
                                                                                                      extensionProperty.extensionName,
                                                                                                      requiredExtension) ==
                                                                                                     0;
                                                                                          });
                                                          });

        if (unsupportedPropertyIt != requiredExtensions.end()) {
            throw std::runtime_error("Required extension not supported: " + std::string(*unsupportedPropertyIt));
        }

        // validation layer code
        // vk_layer_settings.txt
        std::vector<char const *> requiredLayers;
        if (enableValidationLayers) { // NOLINT
            requiredLayers.assign(validationLayers.begin(), validationLayers.end());
        }

        // Check if required layers are supported by the Vulkan implementation
        auto layerProperties = context.enumerateInstanceLayerProperties();
        auto unsupportedLayerIt = std::ranges::find_if(requiredLayers, [&layerProperties](auto const &requiredLayer) {
            return std::ranges::none_of(layerProperties, [requiredLayer](auto const &layerProperty) {
                return strcmp(layerProperty.layerName, requiredLayer) == 0;
            });
        });

        if (unsupportedLayerIt != requiredLayers.end()) {
            throw std::runtime_error("Required layer not supported: " + std::string(*unsupportedLayerIt));
        }

        vk::InstanceCreateInfo createInfo{
                // flag for MoltenVK incompatible driver error
                .flags = vk::InstanceCreateFlagBits::eEnumeratePortabilityKHR,
                .pApplicationInfo = &appInfo,
                .enabledLayerCount = static_cast<uint32_t>(requiredLayers.size()),
                .ppEnabledLayerNames = requiredLayers.data(),
                .enabledExtensionCount = static_cast<uint32_t>(requiredExtensions.size()),
                .ppEnabledExtensionNames = requiredExtensions.data(),
        };

        instance = vk::raii::Instance(context, createInfo);
    }

    std::vector<const char *> static getRequiredInstanceExtensions() {
        uint32_t glfwExtensionCount = 0;
        auto glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

        std::vector extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);
        if (enableValidationLayers) { // NOLINT
            // enable debug messenger with callback
            extensions.push_back(vk::EXTDebugUtilsExtensionName);
        }
        return extensions;
    }

    static VKAPI_ATTR vk::Bool32 VKAPI_CALL debugCallback(vk::DebugUtilsMessageSeverityFlagBitsEXT severity,
                                                          vk::DebugUtilsMessageTypeFlagsEXT type,
                                                          const vk::DebugUtilsMessengerCallbackDataEXT *pCallbackData,
                                                          void *pUserData) {
        std::cerr << "validation layer: type" << to_string(type) << " msg: " << pCallbackData->pMessage << std::endl;

        return vk::False;
    }

    void pickPhysicalDevice() {
        // listing graphics cards
        std::vector<vk::raii::PhysicalDevice> physicalDevices = instance.enumeratePhysicalDevices();
        auto const devIter = std::ranges::find_if(physicalDevices,
                                                  [&](auto const &physDevice) { return isDeviceSuitable(physDevice); });
        if (devIter == physicalDevices.end()) {
            throw std::runtime_error("failed to find a suitable GPU!");
        }
        physicalDevice = *devIter;

        /*
        if (physicalDevices.empty()) {
            throw std::runtime_error("failed to find GPUs with Vulkan support!");
        }

        std::multimap<int, vk::raii::PhysicalDevice> candidates;

        for (const auto& pd: physicalDevices) {
            auto deviceProperties = pd.getProperties();
            auto deviceFeatures = pd.getFeatures();
            uint32_t score = 0;

            if (deviceProperties.deviceType == vk::PhysicalDeviceType::eDiscreteGpu) {
                score += 1000;
            }

            // max possible size of textures affects graphics quality
            score += deviceProperties.limits.maxImageDimension2D;

            if (!deviceFeatures.geometryShader) {
                continue;
            }

            candidates.insert(std::make_pair(score, pd));
        }

        if (!candidates.empty() && candidates.rbegin()->first > 0) {
            physicalDevice = candidates.rbegin()->second;
        } else {
            throw std::runtime_error("failed to find a suitable GPU!");
        }
         */
    }

    std::vector<const char *> requiredDeviceExtension = {vk::KHRSwapchainExtensionName,
            // Required for MoltenVK
                                                         vk::KHRPortabilitySubsetExtensionName,
                                                         vk::KHRSynchronization2ExtensionName};

    bool isDeviceSuitable(vk::raii::PhysicalDevice const &physDevice) {
        auto deviceProperties = physDevice.getProperties();
        // optional features
        auto deviceFeatures = physDevice.getFeatures();

        if (deviceProperties.deviceType == vk::PhysicalDeviceType::eDiscreteGpu && deviceFeatures.geometryShader) {
            return true;
        }

        bool supportsVulkan1_4 = physDevice.getProperties().apiVersion >= vk::ApiVersion14;

        // look for a queue that supports graphics commands
        auto queueFamilies = physDevice.getQueueFamilyProperties();
        bool supportsGraphics = std::ranges::any_of(queueFamilies, [](auto const &qfp) {
            return !!(qfp.queueFlags & vk::QueueFlagBits::eGraphics);
        }); // NOLINT

        // Check if all required physicalDevice extensions
        auto availableDeviceExtensions = physDevice.enumerateDeviceExtensionProperties();
        bool supportsAllRequiredExtensions = std::ranges::all_of(requiredDeviceExtension, [&availableDeviceExtensions](
                auto const &requiredDeviceExtension) {
            return std::ranges::any_of(availableDeviceExtensions,
                                       [requiredDeviceExtension](auto const &availableDeviceExtension) {
                                           return strcmp(availableDeviceExtension.extensionName,
                                                         requiredDeviceExtension) == 0;
                                       });
        });

        // check if the physicalDevice supports the required features (shader drawers, dynamic rendering and extended dynamic state)
        auto features = physDevice.template getFeatures2<vk::PhysicalDeviceFeatures2,
                vk::PhysicalDeviceVulkan11Features,
                vk::PhysicalDeviceVulkan13Features,
                vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>();

        bool supportsRequiredFeatures =
                features.template get<vk::PhysicalDeviceVulkan11Features>().shaderDrawParameters &&
                features.template get<vk::PhysicalDeviceFeatures2>().features.samplerAnisotropy &&
                features.template get<vk::PhysicalDeviceVulkan13Features>().dynamicRendering &&
                features.template get<vk::PhysicalDeviceVulkan13Features>().synchronization2 &&
                features.template get<vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>().extendedDynamicState;

        return supportsVulkan1_4 && supportsGraphics && supportsAllRequiredExtensions && supportsRequiredFeatures;
    }

    void createLogicalDevice() {
        std::vector<vk::QueueFamilyProperties> queueFamilyProperties = physicalDevice.getQueueFamilyProperties();
        // check that logical device can present images to our created window surface
        for (uint32_t qfpIndex = 0; qfpIndex < queueFamilyProperties.size(); qfpIndex++) {
            if ((queueFamilyProperties[qfpIndex].queueFlags & vk::QueueFlagBits::eGraphics) &&
                physicalDevice.getSurfaceSupportKHR(qfpIndex, *surface)) {
                // found a queue family that supports both graphics and present
                queueIndex = qfpIndex;
                break;
            }
        }
        if (queueIndex == ~0) {
            throw std::runtime_error("Could not find a queue for graphics and present -> terminating");
        }

        // Create a chain of feature structures
        vk::StructureChain<vk::PhysicalDeviceFeatures2,
                vk::PhysicalDeviceVulkan11Features,
                vk::PhysicalDeviceVulkan13Features,
                vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>
                featureChain = {
                {.features = {.samplerAnisotropy = true}},              // vk::PhysicalDeviceFeatures2
                {.shaderDrawParameters = true},                         // Enable shader draw parameters from Vulkan 1.1
                {.synchronization2 = true, .dynamicRendering=true},     // Enable dynamic rendering from Vulkan 1.3
                {.extendedDynamicState = true}                          // Enable extended dynamic state from the extension
        };

        // Create Devive + Queue
        float queuePriority = 0.5f;
        vk::DeviceQueueCreateInfo deviceQueueCreateInfo{.queueFamilyIndex = queueIndex, .queueCount=1, .pQueuePriorities = &queuePriority};
        vk::DeviceCreateInfo deviceCreateInfo{
                .pNext = &featureChain.get<vk::PhysicalDeviceFeatures2>(),
                .queueCreateInfoCount = 1,
                .pQueueCreateInfos = &deviceQueueCreateInfo,
                .enabledExtensionCount = static_cast<uint32_t>(requiredDeviceExtension.size()),
                .ppEnabledExtensionNames = requiredDeviceExtension.data(),
        };

        device = vk::raii::Device(physicalDevice, deviceCreateInfo);
        queue = vk::raii::Queue(device, queueIndex, 0);


    }

    // Swap chain utils
    static vk::SurfaceFormatKHR chooseSwapSurfaceFormat(std::vector<vk::SurfaceFormatKHR> const &availableFormats) {
        assert(!availableFormats.empty());
        // opt for SRGB
        const auto formatIt = std::ranges::find_if(
                availableFormats,
                [](const auto &format) {
                    return format.format == vk::Format::eB8G8R8A8Srgb &&
                           format.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear;
                }
        );

        return formatIt != availableFormats.end() ? *formatIt : availableFormats[0];
    }

    static vk::PresentModeKHR chooseSwapPresentMode(std::vector<vk::PresentModeKHR> const &availablePresentModes) {
        assert(std::ranges::any_of(availablePresentModes,
                                   [](auto presentMode) { return presentMode == vk::PresentModeKHR::eFifo; }));
        return std::ranges::any_of(availablePresentModes,
                                   [](const vk::PresentModeKHR value) { return vk::PresentModeKHR::eMailbox == value; })
               ?
               vk::PresentModeKHR::eMailbox :
               vk::PresentModeKHR::eFifo;
    }

    /*
     * Swap extent is resolution of swap chain images, almost always exactly equal to window pixel resolution
     */
    vk::Extent2D chooseSwapExtent(vk::SurfaceCapabilitiesKHR const &capabilities) {
        if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
            return capabilities.currentExtent;
        }
        int width, height;
        // get width/height in pixel, not screen coordinates
        glfwGetFramebufferSize(window, &width, &height);

        return {
                std::clamp<uint32_t>(width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width),
                std::clamp<uint32_t>(height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height)
        };
    }

    void createSwapChain() {
        // 2d resolution of images inside swapchain
        vk::SurfaceCapabilitiesKHR surfaceCapabilities = physicalDevice.getSurfaceCapabilitiesKHR(*surface);
        swapChainExtent = chooseSwapExtent(surfaceCapabilities);
        uint32_t minImageCount = chooseSwapMinImageCount(surfaceCapabilities);

        std::vector<vk::SurfaceFormatKHR> availableFormats = physicalDevice.getSurfaceFormatsKHR(*surface);
        swapChainSurfaceFormat = chooseSwapSurfaceFormat(availableFormats);

        // Supported surface formats
        std::vector<vk::PresentModeKHR> availablePresentModes = physicalDevice.getSurfacePresentModesKHR(*surface);

        vk::SwapchainCreateInfoKHR swapchainCreateInfo{
                .surface=*surface,
                .minImageCount = minImageCount,
                .imageFormat = swapChainSurfaceFormat.format,
                .imageColorSpace = swapChainSurfaceFormat.colorSpace,
                .imageExtent = swapChainExtent,
                .imageArrayLayers = 1,
                .imageUsage = vk::ImageUsageFlagBits::eColorAttachment,
                .imageSharingMode = vk::SharingMode::eExclusive,
                .preTransform = surfaceCapabilities.currentTransform,
                .compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque,
                .presentMode = chooseSwapPresentMode(availablePresentModes),
                .clipped = true
        };
        swapChain = vk::raii::SwapchainKHR(device, swapchainCreateInfo);
        swapChainImages = swapChain.getImages();
    }

    // Image views
    void createImageViews() {
        assert(swapChainImageViews.empty());

        swapChainImageViews.reserve(swapChainImages.size());

        // no components - default color channel mapping: no-swizzling
        for (auto &image: swapChainImages) {
            swapChainImageViews.emplace_back(createImageView(image, swapChainSurfaceFormat.format));
        }
    }

    vk::raii::ImageView createImageView(vk::Image const& image, vk::Format format) {
        vk::ImageViewCreateInfo viewInfo{
            .image = image,
            // TODO: change to flexible 2D/3D
            .viewType = vk::ImageViewType::e2D,
            .format = format,
            .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}
        };

        return {device, viewInfo};
    }

    static uint32_t chooseSwapMinImageCount(vk::SurfaceCapabilitiesKHR const &surfaceCapabilities) {
        auto minImageCount = std::max(3u, surfaceCapabilities.minImageCount);
        if ((0 < surfaceCapabilities.maxImageCount) && (surfaceCapabilities.maxImageCount < minImageCount)) {
            minImageCount = surfaceCapabilities.maxImageCount;
        }
        return minImageCount;
    }

    void createGraphicsPipeline() {
        auto shaderCode = readFile("shaders/slang.spv");
        vk::raii::ShaderModule shaderModule = createShaderModule(shaderCode);

        // pSpecializationInfo to specify values for  runtime shader constants, but more preferable to inject constants on shaderModule creation
        vk::PipelineShaderStageCreateInfo vertShaderStageInfo{.stage=vk::ShaderStageFlagBits::eVertex, .module=shaderModule, .pName="vertMain"};
        vk::PipelineShaderStageCreateInfo fragShaderStageInfo{.stage = vk::ShaderStageFlagBits::eFragment, .module = shaderModule, .pName = "fragMain"};
        vk::PipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo, fragShaderStageInfo};

        auto bindingDescription = Vertex::getBindingDescription();
        auto attributeDescriptions = Vertex::getAttributeDescriptions();

        vk::PipelineVertexInputStateCreateInfo vertexInputInfo{
            .vertexBindingDescriptionCount = 1,
            .pVertexBindingDescriptions = &bindingDescription,
            .vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size()),
            .pVertexAttributeDescriptions = attributeDescriptions.data()
        };
        vk::PipelineInputAssemblyStateCreateInfo inputAssembly{.topology = vk::PrimitiveTopology::eTriangleList};
        vk::PipelineViewportStateCreateInfo viewportState{.viewportCount = 1, .scissorCount = 1};

        vk::PipelineRasterizationStateCreateInfo rasterizer{
                .depthClampEnable = vk::False,
                .rasterizerDiscardEnable=vk::False,
                .polygonMode=vk::PolygonMode::eFill,
                .cullMode = vk::CullModeFlagBits::eNone,
//                .cullMode=vk::CullModeFlagBits::eBack,
                // CCW to counteract backface culling after Y-flip
                .frontFace=vk::FrontFace::eCounterClockwise,
                .depthBiasEnable=vk::False,
                .lineWidth=1.0f
        };

        vk::PipelineMultisampleStateCreateInfo multisampling{.rasterizationSamples=vk::SampleCountFlagBits::e1, .sampleShadingEnable = vk::False};

        vk::PipelineColorBlendAttachmentState colorBlendAttachment{
                .blendEnable=vk::False,
                .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
                                  vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA
        };

        vk::PipelineColorBlendStateCreateInfo colorBlending{
                .logicOpEnable=vk::False, .logicOp = vk::LogicOp::eCopy, .attachmentCount = 1, .pAttachments = &colorBlendAttachment
        };

        std::vector<vk::DynamicState> dynamicStates = {vk::DynamicState::eViewport, vk::DynamicState::eScissor};
        vk::PipelineDynamicStateCreateInfo dynamicState{.dynamicStateCount=static_cast<uint32_t>(dynamicStates.size()), .pDynamicStates = dynamicStates.data()};

        vk::PipelineLayoutCreateInfo pipelineLayoutInfo{.setLayoutCount=1, .pSetLayouts = &*descriptorSetLayout, .pushConstantRangeCount=0 };

        pipelineLayout = vk::raii::PipelineLayout(device, pipelineLayoutInfo);

        // dynamic rendering attachments
        vk::StructureChain<vk::GraphicsPipelineCreateInfo, vk::PipelineRenderingCreateInfo> pipelineCreateInfoChain = {
                {
                        .stageCount = 2,
                        .pStages = shaderStages,
                        .pVertexInputState = &vertexInputInfo,
                        .pInputAssemblyState = &inputAssembly,
                        .pViewportState = &viewportState,
                        .pRasterizationState = &rasterizer,
                        .pMultisampleState = &multisampling,
                        .pColorBlendState = &colorBlending,
                        .pDynamicState = &dynamicState,
                        .layout = pipelineLayout,
                        .renderPass = nullptr
                },
                {.colorAttachmentCount=1, .pColorAttachmentFormats = &swapChainSurfaceFormat.format}
        };

        puts("Attempt graphics pipeline creation");

        graphicsPipeline = vk::raii::Pipeline(device, nullptr,
                                              pipelineCreateInfoChain.get<vk::GraphicsPipelineCreateInfo>());

        puts("Graphics pipeline created");
    }

    static std::vector<char> readFile(const std::string &filename) {
        // start at end of file to get file size for buffer
        std::ifstream file(filename, std::ios::ate | std::ios::binary);

        if (!file.is_open()) {
            std::cerr << "Looking in CWD: " << std::filesystem::current_path() << std::endl;
            throw std::runtime_error("failed to open file!: " + filename);
        }

        std::vector<char> buffer(file.tellg());
        // seek to beginning of file
        file.seekg(0, std::ios::beg);
        file.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        file.close();

        return buffer;
    }

    [[nodiscard]] vk::raii::ShaderModule createShaderModule(const std::vector<char> &code) const {
        // take in buffer with bytecode as parameter and create vk::raii::ShaderModule from it
        vk::ShaderModuleCreateInfo createInfo{
                .codeSize = code.size() * sizeof(char),
                .pCode = reinterpret_cast<const uint32_t *>(code.data())
        };

        vk::raii::ShaderModule shaderModule{device, createInfo};
        return shaderModule;
    }

    void createCommandPool() {
        vk::CommandPoolCreateInfo poolInfo{
                .flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
                .queueFamilyIndex = queueIndex
        };

        commandPool = vk::raii::CommandPool(device, poolInfo);
    }

    void createTextureImages() {
        // TODO: 2D/3D flexibility
        // TODO: populate initial solver data texture here
        // velocity, from solver
        std::vector<float> field_quantity(texHeight * texWidth);
        for (int i = 0; i < texHeight * texWidth; i++) {
            field_quantity[i] = (sin(i) + 1) * 0.5;
        }

        vk::DeviceSize imageSize = field_quantity.size() * sizeof(float);

        for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            auto [textureBuffer, textureBufferMemory] = createBuffer(imageSize, vk::BufferUsageFlagBits::eTransferSrc, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
            void* data = textureBufferMemory.mapMemory(0, imageSize);
            memcpy(data, field_quantity.data(), imageSize);
            // keep memory mapped for updates

            auto [textureImage, textureImageMem] = createImage(
                    texWidth,
                    texHeight,
                    vk::Format::eR32Sfloat,
                    vk::ImageTiling::eOptimal,
                    vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled,
                    vk::MemoryPropertyFlagBits::eDeviceLocal
            );

            vk::raii::CommandBuffer commandBuffer = beginSingleTimeCommands();
            transitionImageLayout(commandBuffer, textureImage, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal);
            copyBufferToImage(commandBuffer, textureBuffer, textureImage, static_cast<uint32_t>(texWidth), static_cast<uint32_t>(texHeight));
            transitionImageLayout(commandBuffer, textureImage, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal);
            endSingleTimeCommands(std::move(commandBuffer));

            textureImageBuffers.emplace_back(std::move(textureBuffer));
            textureImageBuffersMemory.emplace_back(std::move(textureBufferMemory));
            textureImageBuffersMapped.emplace_back(data);

            textureImages.emplace_back(std::move(textureImage));
            textureImagesMemory.emplace_back(std::move(textureImageMem));
        }
    }

    // run after every iteration to pull texture data, particularly scalar field
    void updateSolverImageTextures(uint32_t currentImage) {
        // fetch data
        auto& commandBuffer = commandBuffers[currentImage];
        auto& textureImage = textureImages[currentImage];
        auto& textureImageBuffer = textureImageBuffers[currentImage];
        // TODO: command buffer begin single commands?

        vk::CommandBufferBeginInfo beginInfo {.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit};
        commandBuffer.begin(beginInfo);

        transitionImageLayout(commandBuffer, textureImage, vk::ImageLayout::eShaderReadOnlyOptimal,
                              vk::ImageLayout::eTransferDstOptimal);
        std::vector<float> field_quantity = fluidSolver.get_data();
        std::cerr << "DOBBY\n";

        // find downsampling/interpolation scheme if memory constraints get bad
        memcpy(textureImageBuffersMapped[currentImage], field_quantity.data(), field_quantity.size() * sizeof(float));
        copyBufferToImage(commandBuffer, textureImageBuffer, textureImage, static_cast<uint32_t>(texWidth),
                          static_cast<uint32_t>(texHeight));
        transitionImageLayout(commandBuffer, textureImage, vk::ImageLayout::eTransferDstOptimal,
                              vk::ImageLayout::eShaderReadOnlyOptimal);

        // TODO: consolidate into endSingleCommandsCall
        commandBuffer.end();
        vk::SubmitInfo submitInfo{.commandBufferCount = 1, .pCommandBuffers = &*commandBuffer};
        queue.submit(submitInfo, nullptr);
        queue.waitIdle();
    }

    // image view and sampler
    void createTextureImageViews() {
        // based on createImageViews, only changing format and image
        for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            auto &textureImage = textureImages[i];
            textureImageViews.emplace_back(createImageView(textureImage, vk::Format::eR32Sfloat));
        }
    }

    void createTextureSampler() {
        vk::PhysicalDeviceProperties properties = physicalDevice.getProperties();

        vk::SamplerCreateInfo samplerInfo {
            .magFilter = vk::Filter::eLinear,
            .minFilter = vk::Filter::eLinear,
            .mipmapMode = vk::SamplerMipmapMode::eLinear,
            // TODO: do not use repeat, since we are using a scalar field. Maybe clamp to border?
            // shouldn't matter since we shouldn't sample outside texture
            .addressModeU = vk::SamplerAddressMode::eClampToBorder,
            .addressModeV = vk::SamplerAddressMode::eClampToBorder,
            .addressModeW = vk::SamplerAddressMode::eClampToBorder,
            .anisotropyEnable = vk::True,
            .maxAnisotropy = properties.limits.maxSamplerAnisotropy,
            .compareEnable = vk::False,
            .compareOp = vk::CompareOp::eAlways,
            .borderColor = vk::BorderColor::eFloatOpaqueWhite,
        };

        // no border color, since scalar field. I assume it doesn't work together

        // address to [0,1) instead of unnormalized [0, texHeight/texWidth)
        samplerInfo.unnormalizedCoordinates = vk::False;

        samplerInfo.compareEnable = vk::False;
        samplerInfo.compareOp = vk::CompareOp::eAlways;

        // mipmapping
        samplerInfo.mipmapMode = vk::SamplerMipmapMode::eLinear;
        samplerInfo.mipLodBias = 0.0f;
        samplerInfo.minLod = 0.0f;
        samplerInfo.maxLod = 0.0f;

        // disable anisotropy
        // samplerInfo.anisotropyEnable = vk::False;
        // samplerInfo.maxAnisotropy = 1.0f;

        textureSampler = vk::raii::Sampler(device, samplerInfo);
    }

    std::pair<vk::raii::Image, vk::raii::DeviceMemory> createImage(
            uint32_t width, uint32_t height, vk::Format format, vk::ImageTiling tiling, vk::ImageUsageFlags usage, vk::MemoryPropertyFlags properties
            ) {
        vk::ImageCreateInfo imageInfo {
            // TODO: change to 3D based in dim requested
            .imageType = vk::ImageType::e2D,
            .format = format,
            .extent = {width,height,1},
            .mipLevels = 1,
            .arrayLayers = 1,
            .samples = vk::SampleCountFlagBits::e1,
            .tiling = tiling,
            .usage = usage,
            .sharingMode = vk::SharingMode::eExclusive
        };

        vk::raii::Image image = vk::raii::Image(device, imageInfo);

        vk::MemoryRequirements memRequirements = image.getMemoryRequirements();
        vk::MemoryAllocateInfo allocInfo {
            .allocationSize = memRequirements.size,
            .memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties)
        };
        vk::raii::DeviceMemory imageMemory = vk::raii::DeviceMemory(device, allocInfo);
        image.bindMemory(imageMemory, 0);

        return {std::move(image), std::move(imageMemory)};
    }

    vk::raii::CommandBuffer beginSingleTimeCommands() {
        vk::CommandBufferAllocateInfo allocInfo{
            .commandPool = commandPool,
            .level = vk::CommandBufferLevel::ePrimary,
            .commandBufferCount = 1
        };

        vk::raii::CommandBuffer commandBuffer = std::move(vk::raii::CommandBuffers(device, allocInfo).front());

        vk::CommandBufferBeginInfo beginInfo {.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit};
        commandBuffer.begin(beginInfo);

        return std::move(commandBuffer);
    }

    void endSingleTimeCommands(vk::raii::CommandBuffer&& commandBuffer) {
        commandBuffer.end();

        vk::SubmitInfo submitInfo {
            .commandBufferCount = 1,
            .pCommandBuffers = &*commandBuffer,
        };
        queue.submit(submitInfo, nullptr);
        queue.waitIdle();
    }

    void createCommandBuffers() {
        commandBuffers.clear();
        vk::CommandBufferAllocateInfo allocInfo{.commandPool = commandPool, .level = vk::CommandBufferLevel::ePrimary, .commandBufferCount = MAX_FRAMES_IN_FLIGHT};
        commandBuffers = vk::raii::CommandBuffers(device, allocInfo);
    }

    void recordCommandBuffer(uint32_t imageIndex) {
        // writes commands we want to execute into command buffer
        auto& commandBuffer = commandBuffers[frameIndex];
        commandBuffer.begin({.flags = vk::Flags(vk::CommandBufferUsageFlagBits::eOneTimeSubmit)});

        // Before starting rendering, transition the swapchain image to vk::ImageLayout::eColorAttachmentOptimal
        transition_image_layout(
                imageIndex,
                vk::ImageLayout::eUndefined,
                vk::ImageLayout::eColorAttachmentOptimal,
                {}, // no need to wait for previous operations
                vk::AccessFlagBits2::eColorAttachmentWrite,
                vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                vk::PipelineStageFlagBits2::eColorAttachmentOutput
        );

        // set up color attachment
        vk::ClearValue clearColor = vk::ClearColorValue(0.0f, 0.0f, 0.05f, 1.0f);
        vk::RenderingAttachmentInfo attachmentInfo = {
                .imageView = swapChainImageViews[imageIndex],               // which image view to render to
                .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,    // which layout image will be in during rendering
                .loadOp = vk::AttachmentLoadOp::eClear,                     // what to do with image before render (clear to black beforehand)
                .storeOp = vk::AttachmentStoreOp::eStore,                   // store image for later use
                .clearValue = clearColor
        };

        vk::RenderingInfo renderingInfo = {
                .renderArea = {.offset = {0, 0}, .extent = swapChainExtent},
                .layerCount = 1,
                .colorAttachmentCount = 1,
                .pColorAttachments = &attachmentInfo
        };

        commandBuffers[frameIndex].beginRendering(renderingInfo);

        // bind descriptor sets
        commandBuffers[frameIndex].bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, *descriptorSets[frameIndex], nullptr);

        // drawing
        commandBuffers[frameIndex].bindPipeline(vk::PipelineBindPoint::eGraphics, *graphicsPipeline);

        commandBuffers[frameIndex].setViewport(0, vk::Viewport(0.0f, 0.0f, static_cast<float>(swapChainExtent.width),
                                                  static_cast<float>(swapChainExtent.height)));
        commandBuffers[frameIndex].setScissor(0, vk::Rect2D(vk::Offset2D(0, 0), swapChainExtent));

        // vertex, index buffer
        commandBuffer.bindVertexBuffers(0, *vertexBuffer, {0});
        commandBuffer.bindIndexBuffer(*indexBuffer,0,vk::IndexTypeValue<decltype(indices)::value_type>::value);
        commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, *descriptorSets[frameIndex], nullptr);
        commandBuffer.drawIndexed(static_cast<uint32_t>(indices.size()), 1,0,0,0);
        commandBuffers[frameIndex].endRendering();

        // after rendering, transition the swapchain image to vk::ImageLayout::ePresentSrcKHR
        transition_image_layout(
                imageIndex,
                vk::ImageLayout::eColorAttachmentOptimal,
                vk::ImageLayout::ePresentSrcKHR,
                vk::AccessFlagBits2::eColorAttachmentWrite,
                {},
                vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                vk::PipelineStageFlagBits2::eBottomOfPipe
        );

        commandBuffers[frameIndex].end();
    }

    // sets pipelineBarrier for buffer transfers
    void transitionImageLayout(vk::raii::CommandBuffer& commandBuffer, const vk::raii::Image& image, vk::ImageLayout oldLayout, vk::ImageLayout newLayout) {
        vk::ImageMemoryBarrier barrier{
            .oldLayout = oldLayout,
            .newLayout = newLayout,
            .srcQueueFamilyIndex = vk::QueueFamilyIgnored,
            .dstQueueFamilyIndex = vk::QueueFamilyIgnored,
            .image = image,
            .subresourceRange = {.aspectMask = vk::ImageAspectFlagBits::eColor, .levelCount = 1, .layerCount = 1}
        };
        vk::PipelineStageFlags sourceStage;
        vk::PipelineStageFlags destinationStage;

        if (oldLayout == vk::ImageLayout::eUndefined && newLayout == vk::ImageLayout::eTransferDstOptimal) {
            // does not need any waiting
            barrier.srcAccessMask = {};
            barrier.dstAccessMask = vk::AccessFlagBits::eTransferWrite;

            sourceStage = vk::PipelineStageFlagBits::eTopOfPipe; // pre-barrier
            destinationStage = vk::PipelineStageFlagBits::eTransfer;
        }
        else if (oldLayout == vk::ImageLayout::eTransferDstOptimal && newLayout == vk::ImageLayout::eShaderReadOnlyOptimal) {
            // shader reads should wait on transfer writes - blocking
            barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
            barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;

            sourceStage = vk::PipelineStageFlagBits::eTransfer;
            destinationStage = vk::PipelineStageFlagBits::eFragmentShader;
        }
        else if (oldLayout == vk::ImageLayout::eShaderReadOnlyOptimal && newLayout == vk::ImageLayout::eTransferDstOptimal) {
            // for frame-based updates
            barrier.srcAccessMask = vk::AccessFlagBits::eShaderRead;
            barrier.dstAccessMask = vk::AccessFlagBits::eTransferWrite;

            sourceStage = vk::PipelineStageFlagBits::eFragmentShader;
            destinationStage = vk::PipelineStageFlagBits::eTransfer;
        }
        else {
            std::cerr << std::format("old layout: {}, new layout: {}\n", to_string(oldLayout), to_string(newLayout));
            throw std::invalid_argument("unsupported layout transition");
        }
        commandBuffer.pipelineBarrier(sourceStage, destinationStage, {}, {}, {}, barrier);
    }

    void copyBufferToImage(vk::raii::CommandBuffer& commandBuffer, const vk::raii::Buffer& buffer, vk::raii::Image& image, uint32_t width, uint32_t height) {
        vk::BufferImageCopy region{
            .bufferOffset = 0,
            .bufferRowLength = 0,
            .bufferImageHeight = 0,
            .imageSubresource = {
                    .aspectMask = vk::ImageAspectFlagBits::eColor,
                    .mipLevel = 0,
                    .baseArrayLayer = 0,
                    .layerCount = 1
            },
            .imageOffset = {0,0,0},
            .imageExtent = {width, height, 1}
        };
        commandBuffer.copyBufferToImage(buffer, image, vk::ImageLayout::eTransferDstOptimal, region);
    }

    void transition_image_layout(
            uint32_t imageIndex,
            vk::ImageLayout old_layout,
            vk::ImageLayout new_layout,
            vk::AccessFlags2 src_access_mask,
            vk::AccessFlags2 dst_access_mask,
            vk::PipelineStageFlags2 src_stage_mask,
            vk::PipelineStageFlags2 dst_stage_mask
    ) {
        // pipeline barrier to transition image layout from vk::ImageLayout::eUndefined to vk::ImageLayout::eColorAttachmentOptimal
        vk::ImageMemoryBarrier2 barrier{
                .srcStageMask = src_stage_mask,
                .srcAccessMask = src_access_mask,
                .dstStageMask = dst_stage_mask,
                .dstAccessMask = dst_access_mask,
                .oldLayout = old_layout,
                .newLayout = new_layout,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = swapChainImages[imageIndex],
                .subresourceRange = {
                        .aspectMask = vk::ImageAspectFlagBits::eColor,
                        .baseMipLevel = 0,
                        .levelCount = 1,
                        .baseArrayLayer = 0,
                        .layerCount = 1
                }
        };

        vk::DependencyInfo dependency_info = {
                .dependencyFlags = {},
                .imageMemoryBarrierCount = 1,
                .pImageMemoryBarriers = &barrier
        };
        commandBuffers[frameIndex].pipelineBarrier2(dependency_info);
    }

    void createSyncObjects() {
        assert(presentCompleteSemaphores.empty() && renderFinishedSemaphores.empty() && inFlightFences.empty());
        for (size_t i = 0; i < swapChainImages.size(); i++) {
            renderFinishedSemaphores.emplace_back(device, vk::SemaphoreCreateInfo());
        }

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            presentCompleteSemaphores.emplace_back(device, vk::SemaphoreCreateInfo());
            inFlightFences.emplace_back(device, vk::FenceCreateInfo{.flags = vk::FenceCreateFlagBits::eSignaled});
        }
    }

    void drawFrame() {
        // Note: inFlightFences, presentCompleteSemaphores, and commandBuffers are indexed by frameIndex
        // while renderFinishedSemaphores is indexed by frameIndex

        auto fenceResult = device.waitForFences(*inFlightFences[frameIndex], vk::True, UINT64_MAX);
        if (fenceResult != vk::Result::eSuccess) {
            throw std::runtime_error("failed to wait for fence!");
        }
        device.resetFences(*inFlightFences[frameIndex]);

        auto [result, imageIndex] = swapChain.acquireNextImage(UINT64_MAX, *presentCompleteSemaphores[frameIndex], nullptr);

        // only want eSuccess or eSuboptimalKHR for success code
        if (result != vk::Result::eSuccess && result != vk::Result::eSuboptimalKHR) {
            assert(result == vk::Result::eTimeout || result == vk::Result::eNotReady);
            throw std::runtime_error("failed to acquire swap chain image");
        }
        // update textures from solver, before render mode
        updateSolverImageTextures(frameIndex);
        updateUniformBuffer(frameIndex);

        // only reset fence (GPU to CPU) if we are submitting work
        device.resetFences(*inFlightFences[frameIndex]);

        commandBuffers[frameIndex].reset();
        recordCommandBuffer(imageIndex);

        vk::PipelineStageFlags waitDestinationStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput);
        const vk::SubmitInfo submitInfo{
                .waitSemaphoreCount = 1,
                .pWaitSemaphores = &*presentCompleteSemaphores[frameIndex],
                .pWaitDstStageMask = &waitDestinationStageMask,
                .commandBufferCount = 1,
                .pCommandBuffers = &*commandBuffers[frameIndex],
                .signalSemaphoreCount = 1,
                .pSignalSemaphores = &*renderFinishedSemaphores[imageIndex]
        };
        queue.submit(submitInfo, *inFlightFences[frameIndex]);
        // skip subpass dependencies for now, optional: https://docs.vulkan.org/tutorial/latest/03_Drawing_a_triangle/03_Drawing/02_Rendering_and_presentation.html

        // presentation - submit drawn frame back to swap chain to eventually display
        const vk::PresentInfoKHR presentInfoKHR{
                .waitSemaphoreCount = 1,
                .pWaitSemaphores = &*renderFinishedSemaphores[imageIndex],
                .swapchainCount = 1,
                .pSwapchains = &*swapChain,
                .pImageIndices = &imageIndex
        };

        result = queue.presentKHR(presentInfoKHR);

        assert(result == vk::Result::eSuccess);

        frameIndex = (frameIndex + 1) % MAX_FRAMES_IN_FLIGHT;
    }

    void createDescriptorSetLayout() {
        // descriptor sets
        vk::DescriptorSetLayoutBinding uboLayoutBinding{
            .binding = 0,
            .descriptorType = vk::DescriptorType::eUniformBuffer,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eVertex
        };

        vk::DescriptorSetLayoutBinding samplerLayoutBinding{
            .binding = 1,
            .descriptorType = vk::DescriptorType::eCombinedImageSampler,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eFragment
        };

        std::array<vk::DescriptorSetLayoutBinding, 2> bindings = {
            uboLayoutBinding,
            samplerLayoutBinding
        };

        vk::DescriptorSetLayoutCreateInfo layoutInfo{
                // .sType = vk::StructureType::eDescriptorSetLayoutCreateInfo,
                .bindingCount = static_cast<uint32_t>(bindings.size()),
                .pBindings = bindings.data()
        };

        descriptorSetLayout = vk::raii::DescriptorSetLayout(device, layoutInfo);
        std::vector<vk::DescriptorSetLayout> layouts = {descriptorSetLayout};   // change this to std::array later
    }

    std::pair<vk::raii::Buffer, vk::raii::DeviceMemory> createBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage, vk::MemoryPropertyFlags properties) {
        vk::BufferCreateInfo bufferInfo {
            .size=size,
            .usage=usage,
            .sharingMode=vk::SharingMode::eExclusive
        };
        vk::raii::Buffer buffer = vk::raii::Buffer(device, bufferInfo);
        vk::MemoryRequirements memRequirements = buffer.getMemoryRequirements();
        vk::MemoryAllocateInfo allocInfo{.allocationSize = memRequirements.size, .memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties)};
        vk::raii::DeviceMemory bufferMemory = vk::raii::DeviceMemory(device, allocInfo);
        buffer.bindMemory(*bufferMemory, 0);
        // for some reason std::move required here (or non-constructible, maybe missing copy constructor) rather than NRVO
        return {std::move(buffer), std::move(bufferMemory)};
    }

    // Do not need VBO or Index Buffer, but adding just in case
    void createVertexBuffer() {
        vk::DeviceSize bufferSize = sizeof(vertices[0]) * vertices.size();

        auto [stagingBuffer, stagingBufferMemory] = createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferSrc, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);

        void* dataStaging = stagingBufferMemory.mapMemory(0, bufferSize);
        memcpy(dataStaging, vertices.data(), bufferSize);
        // fine to unmap, since we aren't mapping again
        stagingBufferMemory.unmapMemory();

        std::tie(vertexBuffer, vertexBufferMemory) = createBuffer(bufferSize, vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eTransferDst, vk::MemoryPropertyFlagBits::eDeviceLocal);

        copyBuffer(stagingBuffer, vertexBuffer, bufferSize);
    }

    void createIndexBuffer() {
        vk::DeviceSize bufferSize = sizeof(indices[0]) * indices.size();

        auto [stagingBuffer, stagingBufferMemory] = createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferSrc, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);

        void* data = stagingBufferMemory.mapMemory(0, bufferSize);
        memcpy(data, indices.data(), static_cast<size_t>(bufferSize));
        stagingBufferMemory.unmapMemory();

        std::tie(indexBuffer, indexBufferMemory) = createBuffer(bufferSize, vk::BufferUsageFlagBits::eIndexBuffer | vk::BufferUsageFlagBits::eTransferDst, vk::MemoryPropertyFlagBits::eDeviceLocal);

        copyBuffer(stagingBuffer, indexBuffer, bufferSize);
    }

    void copyBuffer(vk::raii::Buffer& srcBuffer, vk::raii::Buffer& dstBuffer, vk::DeviceSize size) {
        vk::raii::CommandBuffer commandCopyBuffer = beginSingleTimeCommands();
        commandCopyBuffer.copyBuffer(*srcBuffer, *dstBuffer, vk::BufferCopy{.size=size});
        endSingleTimeCommands(std::move(commandCopyBuffer));
    }

    void createUniformBuffers() {
        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            vk::DeviceSize bufferSize = sizeof(UniformBufferObject);
            auto [buffer, bufferMem] = createBuffer(bufferSize, vk::BufferUsageFlagBits::eUniformBuffer, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
            uniformBuffers.emplace_back(std::move(buffer));
            uniformBuffersMemory.emplace_back(std::move(bufferMem));
            uniformBuffersMapped.emplace_back(uniformBuffersMemory.back().mapMemory(0, bufferSize));
        }
    }

    void updateUniformBuffer(uint32_t currentImage) {
        static auto startTime = std::chrono::high_resolution_clock::now();

        auto currentTime = std::chrono::high_resolution_clock::now();
        float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime-startTime).count();

        UniformBufferObject ubo{};
        // TODO: populate MVP matrices for 3d simulation

        memcpy(uniformBuffersMapped[currentImage], &ubo, sizeof(ubo));
    }

    void createDescriptorPool() {
        // descriptor sets must be allocated from pool, like command buffers
        vk::DescriptorPoolSize uniformPoolSize {
            .type = vk::DescriptorType::eUniformBuffer,
            .descriptorCount = MAX_FRAMES_IN_FLIGHT
        };

        vk::DescriptorPoolSize combinedImageSamplerPoolSize {
            .type = vk::DescriptorType::eCombinedImageSampler,
            .descriptorCount = MAX_FRAMES_IN_FLIGHT
        };

        std::array<vk::DescriptorPoolSize,2> poolSize = {uniformPoolSize, combinedImageSamplerPoolSize};

        // Note: improper allocation may sometimes throw vk::OutOfPoolMemoryError
        // Not all machines, since some graphics drivers may automatically handle
        // Will be warned if best practice validation enabled for VK_LAYER_KHRONOS_validation
        vk::DescriptorPoolCreateInfo poolInfo{
            .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
            .maxSets = MAX_FRAMES_IN_FLIGHT,
            .poolSizeCount = static_cast<uint32_t>(poolSize.size()),
            .pPoolSizes = poolSize.data()
        };

        descriptorPool = vk::raii::DescriptorPool(device, poolInfo);
    }

    void createDescriptorSets() {
        std::vector<vk::DescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, *descriptorSetLayout);
        vk::DescriptorSetAllocateInfo allocInfo {
            .descriptorPool = descriptorPool,
            .descriptorSetCount = static_cast<uint32_t>(layouts.size()),
            .pSetLayouts = layouts.data()
        };

        descriptorSets = device.allocateDescriptorSets(allocInfo);

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            vk::DescriptorBufferInfo bufferInfo {
                .buffer = uniformBuffers[i],
                .offset = 0,
                .range = sizeof(UniformBufferObject)
            };

            vk::DescriptorImageInfo imageInfo {
                .sampler = textureSampler,
                .imageView = textureImageViews[i],
                .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal
            };

            std::array<vk::WriteDescriptorSet, 2> descriptorWrites{{{
                .dstSet = descriptorSets[i],
                .dstBinding = 0,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = vk::DescriptorType::eUniformBuffer,
                .pBufferInfo = &bufferInfo
            }, {
                .dstSet = descriptorSets[i],
                .dstBinding = 1,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                .pImageInfo = &imageInfo
            }}};

            device.updateDescriptorSets(descriptorWrites,{});
        }
    }

    uint32_t findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties) {
        vk::PhysicalDeviceMemoryProperties memProperties = physicalDevice.getMemoryProperties();

        for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
            if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
                return i;
            }
        }
        throw std::runtime_error("failed to find suitable memory type.");
    }

    SOLVER_T fluidSolver;

    GLFWwindow *window = nullptr;

    vk::raii::Context context;
    vk::raii::Instance instance = nullptr;

    vk::raii::DebugUtilsMessengerEXT debugMessenger = nullptr;

    vk::raii::PhysicalDevice physicalDevice = nullptr;

    vk::raii::Device device = nullptr;
    uint32_t queueIndex = ~0;
    vk::raii::Queue queue = nullptr;

    // window surface creation
    vk::raii::SurfaceKHR surface = nullptr;

    // swap chain
    vk::raii::SwapchainKHR swapChain = nullptr;
    std::vector<vk::Image> swapChainImages;
    vk::SurfaceFormatKHR swapChainSurfaceFormat;
    vk::Extent2D swapChainExtent;

    // image views
    std::vector<vk::raii::ImageView> swapChainImageViews;

    // Descriptor Set
    vk::raii::DescriptorSetLayout descriptorSetLayout = nullptr;

    // Graphics Pipeline
    vk::raii::PipelineLayout pipelineLayout = nullptr;
    vk::raii::Pipeline graphicsPipeline = nullptr;

    // Data Buffers
    vk::raii::Buffer vertexBuffer = nullptr;
    vk::raii::DeviceMemory vertexBufferMemory = nullptr;
    vk::raii::Buffer indexBuffer = nullptr;
    vk::raii::DeviceMemory indexBufferMemory = nullptr;

    // Command Buffers
    vk::raii::CommandPool commandPool = nullptr;
    std::vector<vk::raii::CommandBuffer> commandBuffers;

    // Rendering and presentation
    // semaphore for GPU blocking, fence for CPU (Host) blocking
    std::vector<vk::raii::Semaphore> presentCompleteSemaphores;
    std::vector<vk::raii::Semaphore> renderFinishedSemaphores;
    std::vector<vk::raii::Fence> inFlightFences;
    uint32_t frameIndex = 0;

    std::vector<vk::raii::Buffer> uniformBuffers;
    std::vector<vk::raii::DeviceMemory> uniformBuffersMemory;
    std::vector<void*> uniformBuffersMapped;

    // Image View and Sampler
    // Texture Image (in this case sampler2D, or maybe colormap)
    // https://docs.vulkan.org/tutorial/latest/06_Texture_mapping/00_Images.html
    std::vector<vk::raii::Image> textureImages;
    std::vector<vk::raii::ImageView> textureImageViews;
    vk::raii::Sampler textureSampler = nullptr;
    // Not sure if we want to downsample later for memory constraints
    int texWidth;
    int texHeight;

    vk::raii::DescriptorPool descriptorPool = nullptr;
    std::vector<vk::raii::DescriptorSet> descriptorSets;

    // Staging Buffer for Texture Image
    std::vector<vk::raii::Buffer> textureImageBuffers;
    std::vector<vk::raii::DeviceMemory> textureImageBuffersMemory;
    std::vector<void*> textureImageBuffersMapped;
    std::vector<vk::raii::DeviceMemory> textureImagesMemory;
};

int main() {
    try {
        FluidSimApplication<> app;
        app.run();
    }
    catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}