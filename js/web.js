import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { $el } from "../../../scripts/ui.js";

let extension = {
    name: "FairLab",
    nodeCreated(node, app) {
        if (node?.comfyClass === "DownloadImageNode") {
            node.onExecuted = function (output) {
                let imgURLs = [];
                if (output?.images) {
                    imgURLs = imgURLs.concat(
                        output.images.map((params) => {
                            return api.apiURL("/view?" + new URLSearchParams(params).toString() + (this.animatedImages ? "" : app.getPreviewFormatParam()) + app.getRandParam());
                        })
                    );
                }

                if (imgURLs.length > 0) {
                    Promise.all(
                        imgURLs.map((src) => {
                            return new Promise((r) => {
                                const img = new Image();
                                img.onload = () => r(img);
                                img.onerror = () => r(null);
                                img.src = src;
                            });
                        })
                    ).then((imgs) => {
                        function download_images(imgs) {
                            for (let i = 0; i < imgs.length; i++) {
                                let img = imgs[i];
                                let a = document.createElement("a");
                                let url = new URL(img.src);
                                url.searchParams.delete("preview");
                                a.href = url;
                                a.setAttribute("download", new URLSearchParams(url.search).get("filename"));
                                document.body.append(a);
                                a.click();
                                requestAnimationFrame(() => a.remove());
                            }
                        }

                        if (imgs) download_images(imgs);
                    });
                }
            };
        }
    },

    async setup() {
        let restartButton;

        //old ui
        const menu = document.querySelector(".comfy-menu");
        if (menu) {
            restartButton = document.createElement("button");
            restartButton.textContent = "Restart";
            restartButton.tooltip = "Restart the server";
            restartButton.onclick = () => {
                api.fetchApi("/manager/reboot");
            };
            menu.appendChild(restartButton);
        }

        //new ui
        if (!app.menu?.element.style.display && app.menu?.settingsGroup) {
            restartButton = new (await import("../../../scripts/ui/components/button.js")).ComfyButton({
                icon: "restart",
                action: () => {
                    api.fetchApi("/manager/reboot");
                },
                tooltip: "Restart the server",
                content: "Restart",
            });
            restartButton.enabled = true;
            restartButton.element.style.display = "";
            app.menu.settingsGroup.append(restartButton);
        }
    },

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (["OllamaNode"].includes(nodeData.name)) {
            const originalNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = async function () {
                if (originalNodeCreated) {
                    originalNodeCreated.apply(this, arguments);
                }

                const urlWidget = this.widgets.find((w) => w.name === "url");
                const modelWidget = this.widgets.find((w) => w.name === "model");

                const fetchModels = async (url) => {
                    try {
                        const response = await fetch("/ollama/get_models", {
                            method: "POST",
                            headers: {
                                "Content-Type": "application/json",
                            },
                            body: JSON.stringify({
                                url,
                            }),
                        });

                        if (response.ok) {
                            const models = await response.json();
                            console.debug("Fetched models:", models);
                            return models;
                        } else {
                            console.error(`Failed to fetch models: ${response.status}`);
                            return [];
                        }
                    } catch (error) {
                        console.error(`Error fetching models`, error);
                        return [];
                    }
                };

                const updateModels = async () => {
                    const url = urlWidget.value;
                    const prevValue = modelWidget.value;
                    modelWidget.value = "";
                    modelWidget.options.values = [];

                    const models = await fetchModels(url);

                    // Update modelWidget options and value
                    modelWidget.options.values = models;
                    console.debug("Updated modelWidget.options.values:", modelWidget.options.values);

                    if (models.includes(prevValue)) {
                        modelWidget.value = prevValue; // stay on current.
                    } else if (models.length > 0) {
                        modelWidget.value = models[0]; // set first as default.
                    }

                    console.debug("Updated modelWidget.value:", modelWidget.value);
                };

                urlWidget.callback = updateModels;

                const dummy = async () => {
                    // calling async method will update the widgets with actual value from the browser and not the default from Node definition.
                };

                // Initial update
                await dummy(); // this will cause the widgets to obtain the actual value from web page.
                await updateModels();
            };
        }
    },
};

app.registerExtension(extension);
