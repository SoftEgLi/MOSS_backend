package config

import (
	"github.com/gofiber/fiber/v2"

	. "MOSS_backend/models"
	. "MOSS_backend/utils"
)

// GetConfig
// @Summary get global config
// @Tags Config
// @Produce json
// @Router /config [get]
// @Success 200 {object} Response
func GetConfig(c *fiber.Ctx) error {
	var configObject Config
	err := LoadConfig(&configObject)
	if err != nil {
		return err
	}

	//var region string
	//ok, err := IsInChina(GetRealIP(c))
	//if err != nil {
	//	return err
	//}
	//if ok {
	//	region = "cn"
	//} else {
	//	region = "global"
	//}
	var region = "global"

	return c.JSON(Response{
		Region:         region,
		InviteRequired: configObject.InviteRequired,
		Notice:         configObject.Notice,
		ModelConfig:    FromModelConfig(configObject.ModelConfig),
	})
}

// PatchConfig
// @Summary update global config
// @Tags Config
// @Accept json
// @Produce json
// @Router /config [patch]
// @Param json body ModifyModelConfigRequest true "body"
// @Success 200 {object} Response
// @Failure 400 {object} Response
// @Failure 500 {object} Response
func PatchConfig(c *fiber.Ctx) error {
	var configObject Config
	err := LoadConfig(&configObject)
	if err != nil {
		return InternalServerError("Failed to load config")
	}

	var body ModifyModelConfigRequest
	err = ValidateBody(c, &body)
	if err != nil {
		return BadRequest(err.Error())
	}

	if body.InviteRequired != nil {
		configObject.InviteRequired = *body.InviteRequired
	}
	if body.OffenseCheck != nil {
		configObject.OffenseCheck = *body.OffenseCheck
	}
	if body.Notice != nil {
		configObject.Notice = *body.Notice
	}
	if body.ModelConfig != nil {
		newModelCfg := body.ModelConfig
		for _, newSingleCfg := range newModelCfg {
			modelID := *(newSingleCfg.ID)
			for i := range configObject.ModelConfig {
				if configObject.ModelConfig[i].ID == modelID {
					if newSingleCfg.Description != nil {
						configObject.ModelConfig[i].Description = *(newSingleCfg.Description)
					}
					if newSingleCfg.InnerThoughtsPostprocess != nil {
						configObject.ModelConfig[i].InnerThoughtsPostprocess = *(newSingleCfg.InnerThoughtsPostprocess)
					}
					if newSingleCfg.DefaultPluginConfig != nil {
						if configObject.ModelConfig[i].DefaultPluginConfig == nil {
							// this means the default plugin config is never set
							configObject.ModelConfig[i].DefaultPluginConfig = *(newSingleCfg.DefaultPluginConfig)
						} else {
							for k, v := range *(newSingleCfg.DefaultPluginConfig) {
								if _, ok := configObject.ModelConfig[i].DefaultPluginConfig[k]; ok {
									configObject.ModelConfig[i].DefaultPluginConfig[k] = v
								}
							}
						}

					}
				}
			}
		}
	}

	// 将更新后的 configObject 保存到数据库中
	err = UpdateConfig(&configObject)
	if err != nil {
		return InternalServerError("Failed to update config")
	}

	return c.Status(200).JSON(fiber.Map{
		"success": "Config updated successfully",
	})
}
