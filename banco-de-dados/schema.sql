-- MySQL dump 10.13  Distrib 8.0.46, for Win64 (x86_64)
--
-- Host: localhost    Database: projeto_sono
-- ------------------------------------------------------
-- Server version	8.0.46

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!50503 SET NAMES utf8mb4 */;
/*!40103 SET @OLD_TIME_ZONE=@@TIME_ZONE */;
/*!40103 SET TIME_ZONE='+00:00' */;
/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;

--
-- Current Database: `projeto_sono`
--

CREATE DATABASE /*!32312 IF NOT EXISTS*/ `projeto_sono` /*!40100 DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci */ /*!80016 DEFAULT ENCRYPTION='N' */;

USE `projeto_sono`;

--
-- Table structure for table `classes_modelo_ml`
--

DROP TABLE IF EXISTS `classes_modelo_ml`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `classes_modelo_ml` (
  `id_classe_modelo` tinyint unsigned NOT NULL AUTO_INCREMENT,
  `taxonomia` varchar(50) NOT NULL,
  `rotulo_modelo` tinyint unsigned NOT NULL,
  `codigo` varchar(20) NOT NULL,
  `nome` varchar(60) NOT NULL,
  `descricao` varchar(255) DEFAULT NULL,
  PRIMARY KEY (`id_classe_modelo`),
  UNIQUE KEY `uq_classes_modelo_taxonomia_rotulo` (`taxonomia`,`rotulo_modelo`),
  UNIQUE KEY `uq_classes_modelo_taxonomia_codigo` (`taxonomia`,`codigo`)
) ENGINE=InnoDB AUTO_INCREMENT=5 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `epocas`
--

DROP TABLE IF EXISTS `epocas`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `epocas` (
  `id_epoca` bigint unsigned NOT NULL AUTO_INCREMENT,
  `id_noite` int unsigned NOT NULL,
  `numero_epoca` smallint unsigned NOT NULL,
  `inicio_utc` datetime(6) NOT NULL,
  `fim_utc` datetime(6) NOT NULL,
  `hr_mean` double NOT NULL,
  `hr_min` double NOT NULL,
  `hr_max` double NOT NULL,
  `hr_std` double NOT NULL,
  `hr_samples` smallint unsigned NOT NULL,
  `motion_mean` double NOT NULL,
  `motion_std` double NOT NULL,
  `motion_max` double NOT NULL,
  `motion_samples` smallint unsigned NOT NULL,
  `duracao_coberta` double NOT NULL,
  `cobertura_temporal_pct` double NOT NULL,
  `maior_gap` double NOT NULL,
  `id_estagio_referencia` tinyint unsigned NOT NULL,
  PRIMARY KEY (`id_epoca`),
  UNIQUE KEY `uk_noite_epoca` (`id_noite`,`numero_epoca`),
  KEY `idx_epoca_estagio` (`id_estagio_referencia`),
  CONSTRAINT `fk_epoca_estagio` FOREIGN KEY (`id_estagio_referencia`) REFERENCES `estagios_sono` (`id_estagio`) ON DELETE RESTRICT ON UPDATE CASCADE,
  CONSTRAINT `fk_epoca_noite` FOREIGN KEY (`id_noite`) REFERENCES `noites` (`id_noite`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB AUTO_INCREMENT=72418 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `estagios_sono`
--

DROP TABLE IF EXISTS `estagios_sono`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `estagios_sono` (
  `id_estagio` tinyint unsigned NOT NULL,
  `codigo` varchar(10) COLLATE utf8mb4_unicode_ci NOT NULL,
  `nome` varchar(50) COLLATE utf8mb4_unicode_ci NOT NULL,
  `descricao` varchar(255) COLLATE utf8mb4_unicode_ci DEFAULT NULL,
  PRIMARY KEY (`id_estagio`),
  UNIQUE KEY `uk_estagio_codigo` (`codigo`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `metricas_noite`
--

DROP TABLE IF EXISTS `metricas_noite`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `metricas_noite` (
  `id_metrica` int unsigned NOT NULL AUTO_INCREMENT,
  `id_noite` int unsigned NOT NULL,
  `duracao_analisada_min` double DEFAULT NULL,
  `tempo_sono_min` double DEFAULT NULL,
  `wake_pct` double DEFAULT NULL,
  `n1_pct` double DEFAULT NULL,
  `n2_pct` double DEFAULT NULL,
  `n3_pct` double DEFAULT NULL,
  `rem_pct` double DEFAULT NULL,
  `hr_media` double DEFAULT NULL,
  `movimento_medio` double DEFAULT NULL,
  `transicoes_estagio` int unsigned DEFAULT NULL,
  `possiveis_despertares` int unsigned DEFAULT NULL,
  `indice_experimental` double DEFAULT NULL,
  PRIMARY KEY (`id_metrica`),
  UNIQUE KEY `uk_metricas_noite` (`id_noite`),
  CONSTRAINT `fk_metricas_noite` FOREIGN KEY (`id_noite`) REFERENCES `noites` (`id_noite`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB AUTO_INCREMENT=76 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `metricas_noite_modelo_ml`
--

DROP TABLE IF EXISTS `metricas_noite_modelo_ml`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `metricas_noite_modelo_ml` (
  `id_metrica_modelo` bigint unsigned NOT NULL AUTO_INCREMENT,
  `id_noite` int unsigned NOT NULL,
  `id_modelo` int unsigned NOT NULL,
  `duracao_analisada_min` double DEFAULT NULL,
  `tempo_sono_estimado_min` double DEFAULT NULL,
  `wake_pct` double DEFAULT NULL,
  `light_pct` double DEFAULT NULL,
  `deep_pct` double DEFAULT NULL,
  `rem_pct` double DEFAULT NULL,
  `hr_media` double DEFAULT NULL,
  `movimento_medio` double DEFAULT NULL,
  `transicoes_estagio` int unsigned DEFAULT NULL,
  `possiveis_despertares` int unsigned DEFAULT NULL,
  `confianca_media` double DEFAULT NULL,
  `indice_experimental` double DEFAULT NULL,
  `atualizado_em` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (`id_metrica_modelo`),
  UNIQUE KEY `uq_metricas_noite_modelo` (`id_noite`,`id_modelo`),
  KEY `idx_metricas_modelo` (`id_modelo`),
  CONSTRAINT `fk_metricas_modelo_modelo` FOREIGN KEY (`id_modelo`) REFERENCES `modelos_ml` (`id_modelo`) ON DELETE CASCADE ON UPDATE CASCADE,
  CONSTRAINT `fk_metricas_modelo_noite` FOREIGN KEY (`id_noite`) REFERENCES `noites` (`id_noite`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB AUTO_INCREMENT=16 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `modelos_ml`
--

DROP TABLE IF EXISTS `modelos_ml`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `modelos_ml` (
  `id_modelo` int unsigned NOT NULL AUTO_INCREMENT,
  `nome` varchar(100) COLLATE utf8mb4_unicode_ci NOT NULL,
  `algoritmo` varchar(100) COLLATE utf8mb4_unicode_ci NOT NULL,
  `versao` varchar(30) COLLATE utf8mb4_unicode_ci NOT NULL,
  `data_treinamento` datetime DEFAULT NULL,
  `accuracy` double DEFAULT NULL,
  `macro_f1` double DEFAULT NULL,
  `observacoes` text COLLATE utf8mb4_unicode_ci,
  `balanced_accuracy` double DEFAULT NULL,
  `weighted_f1` double DEFAULT NULL,
  `taxonomia` varchar(50) COLLATE utf8mb4_unicode_ci DEFAULT NULL,
  `quantidade_classes` tinyint unsigned DEFAULT NULL,
  PRIMARY KEY (`id_modelo`),
  UNIQUE KEY `uk_modelo_versao` (`nome`,`versao`)
) ENGINE=InnoDB AUTO_INCREMENT=3 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `noites`
--

DROP TABLE IF EXISTS `noites`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `noites` (
  `id_noite` int unsigned NOT NULL AUTO_INCREMENT,
  `id_participante` int unsigned NOT NULL,
  `numero_noite` tinyint unsigned NOT NULL,
  `inicio_utc` datetime(6) DEFAULT NULL,
  `fim_utc` datetime(6) DEFAULT NULL,
  `epocas_originais` smallint unsigned DEFAULT NULL,
  `epocas_unknown` smallint unsigned DEFAULT '0',
  `epocas_validas` smallint unsigned DEFAULT NULL,
  `epocas_removidas` smallint unsigned DEFAULT '0',
  `cobertura_percentual` decimal(6,3) DEFAULT NULL,
  PRIMARY KEY (`id_noite`),
  UNIQUE KEY `uk_participante_noite` (`id_participante`,`numero_noite`),
  CONSTRAINT `fk_noite_participante` FOREIGN KEY (`id_participante`) REFERENCES `participantes` (`id_participante`) ON DELETE RESTRICT ON UPDATE CASCADE
) ENGINE=InnoDB AUTO_INCREMENT=91 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `participantes`
--

DROP TABLE IF EXISTS `participantes`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `participantes` (
  `id_participante` int unsigned NOT NULL AUTO_INCREMENT,
  `codigo` varchar(20) COLLATE utf8mb4_unicode_ci NOT NULL,
  PRIMARY KEY (`id_participante`),
  UNIQUE KEY `uk_participante_codigo` (`codigo`)
) ENGINE=InnoDB AUTO_INCREMENT=19 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `predicoes`
--

DROP TABLE IF EXISTS `predicoes`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `predicoes` (
  `id_predicao` bigint unsigned NOT NULL AUTO_INCREMENT,
  `id_epoca` bigint unsigned NOT NULL,
  `id_modelo` int unsigned NOT NULL,
  `id_estagio_previsto` tinyint unsigned NOT NULL,
  `confianca` double DEFAULT NULL,
  PRIMARY KEY (`id_predicao`),
  UNIQUE KEY `uk_predicao_epoca_modelo` (`id_epoca`,`id_modelo`),
  KEY `fk_predicao_modelo` (`id_modelo`),
  KEY `fk_predicao_estagio` (`id_estagio_previsto`),
  CONSTRAINT `fk_predicao_epoca` FOREIGN KEY (`id_epoca`) REFERENCES `epocas` (`id_epoca`) ON DELETE CASCADE ON UPDATE CASCADE,
  CONSTRAINT `fk_predicao_estagio` FOREIGN KEY (`id_estagio_previsto`) REFERENCES `estagios_sono` (`id_estagio`) ON DELETE RESTRICT ON UPDATE CASCADE,
  CONSTRAINT `fk_predicao_modelo` FOREIGN KEY (`id_modelo`) REFERENCES `modelos_ml` (`id_modelo`) ON DELETE RESTRICT ON UPDATE CASCADE
) ENGINE=InnoDB AUTO_INCREMENT=12062 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `predicoes_modelo_ml`
--

DROP TABLE IF EXISTS `predicoes_modelo_ml`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `predicoes_modelo_ml` (
  `id_predicao` bigint unsigned NOT NULL AUTO_INCREMENT,
  `id_epoca` bigint unsigned NOT NULL,
  `id_modelo` int unsigned NOT NULL,
  `id_classe_modelo` tinyint unsigned NOT NULL,
  `confianca` double DEFAULT NULL,
  `criado_em` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id_predicao`),
  UNIQUE KEY `uq_predicoes_modelo_epoca_modelo` (`id_epoca`,`id_modelo`),
  KEY `idx_predicoes_modelo_modelo` (`id_modelo`),
  KEY `idx_predicoes_modelo_classe` (`id_classe_modelo`),
  CONSTRAINT `fk_predicoes_modelo_classe` FOREIGN KEY (`id_classe_modelo`) REFERENCES `classes_modelo_ml` (`id_classe_modelo`) ON DELETE RESTRICT ON UPDATE CASCADE,
  CONSTRAINT `fk_predicoes_modelo_epoca` FOREIGN KEY (`id_epoca`) REFERENCES `epocas` (`id_epoca`) ON DELETE CASCADE ON UPDATE CASCADE,
  CONSTRAINT `fk_predicoes_modelo_modelo` FOREIGN KEY (`id_modelo`) REFERENCES `modelos_ml` (`id_modelo`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB AUTO_INCREMENT=12253 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping events for database 'projeto_sono'
--

--
-- Dumping routines for database 'projeto_sono'
--
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2026-11-24 15:35:09
